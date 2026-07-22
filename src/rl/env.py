"""Gymnasium-compatible wrapper around robosuite's ``Lift`` task (Franka Panda).

robosuite ships its own ``GymWrapper``, but it targets the legacy OpenAI Gym API
(4-tuple ``step``, no ``seed`` / ``options`` on ``reset``). Stable-Baselines3 >= 2.0
speaks *Gymnasium*, so we wrap the raw robosuite environment directly.

Two details matter for correctness and are handled explicitly:

1. ``terminated`` vs ``truncated``. robosuite raises its ``done`` flag when the
   horizon is reached. That is *truncation*, not termination. Reporting it as
   ``terminated`` tells SB3 the value function is zero at the cut-off and
   silently biases every bootstrap target.
2. ``info["is_success"]``. SB3's ``EvalCallback`` aggregates this key into
   ``eval/success_rate``. Without it you only ever see return, never task success.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces

from objects import lift_object_task  # noqa: F401  -- registers "ParamLift" with robosuite
from objects.object_params import ObjectParams

LOGGER = logging.getLogger(__name__)

#: robosuite's low-dim state keys. `robot0_proprio-state` = joint pos/vel, gripper,
#: eef pose. `object-state` = cube pose + relative-to-eef vector.
DEFAULT_OBS_KEYS: tuple[str, ...] = ("robot0_proprio-state", "object-state")


@dataclass
class EnvConfig:
    """Everything needed to construct one :class:`RobosuiteLiftEnv`."""

    task: str = "Lift"
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    horizon: int = 500
    control_freq: int = 20
    #: robosuite's *default* for Lift is sparse (``reward_shaping=False``). Sparse
    #: reward + random exploration on a 7-DoF arm means ~zero successes and no
    #: learning signal, so we default to the shaped reward. Flip to False to
    #: reproduce the true library default.
    reward_shaping: bool = True
    #: End the episode the moment the cube is lifted. Off by default: fixed-length
    #: episodes keep the return comparable across runs.
    terminate_on_success: bool = False
    obs_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_OBS_KEYS)
    render_mode: str | None = None
    camera_name: str = "agentview"
    camera_height: int = 256
    camera_width: int = 256
    #: LLM-derived physical params for the liftable object. None reproduces the stock
    #: robosuite Lift cube exactly (env_name="Lift", no ParamLift involved at all).
    object: ObjectParams | None = None

    def __post_init__(self) -> None:
        # JSON round-trips a tuple back as a list. Normalise on construction so a
        # config and its reloaded snapshot compare equal, and so `rollout.py`
        # rebuilds the exact observation layout the policy was trained on.
        self.obs_keys = tuple(self.obs_keys)
        # Same round-trip concern: a loaded JSON snapshot hands back a plain dict.
        if isinstance(self.object, dict):
            self.object = ObjectParams(**self.object)


def _load_controller_config(controller_name: str, robot: str) -> dict[str, Any]:
    """Return a controller config across the robosuite 1.4 / 1.5 API split.

    robosuite 1.5 replaced ``load_controller_config(default_controller=...)`` with
    composite (per-body-part) controllers. ``"BASIC"`` is the composite equivalent
    of the old ``"OSC_POSE"`` arm controller plus a binary gripper. On 1.5 the
    ``controller`` field in the JSON config is therefore ignored.
    """
    try:
        from robosuite.controllers import load_composite_controller_config
    except ImportError:  # robosuite < 1.5
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller=controller_name)
    return load_composite_controller_config(controller="BASIC", robot=robot)


class RobosuiteLiftEnv(gym.Env):
    """A single-agent, state-observation Gymnasium view of a robosuite task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self._obs_keys = tuple(self.cfg.obs_keys)
        self.render_mode = self.cfg.render_mode

        object_kwargs: dict[str, Any] = {}
        env_name = self.cfg.task
        if self.cfg.object is not None:
            if self.cfg.task != "Lift":
                raise ValueError(
                    f"cfg.object is only supported for task='Lift', got {self.cfg.task!r}"
                )
            env_name = "ParamLift"
            object_kwargs["object_params"] = self.cfg.object

        self._env = suite.make(
            env_name=env_name,
            robots=self.cfg.robot,
            controller_configs=_load_controller_config(self.cfg.controller, self.cfg.robot),
            has_renderer=self.render_mode == "human",
            has_offscreen_renderer=self.render_mode == "rgb_array",
            use_camera_obs=False,  # state-only: keeps the MLP policy small and fast
            use_object_obs=True,
            reward_shaping=self.cfg.reward_shaping,
            horizon=self.cfg.horizon,
            control_freq=self.cfg.control_freq,
            ignore_done=True,  # we own episode termination; see module docstring
            hard_reset=False,  # ~10x faster resets; re-uses the compiled MjModel
            **object_kwargs,
        )

        low, high = self._env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

        obs_dim = self._flatten(self._env.reset()).shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._elapsed_steps = 0
        LOGGER.info(
            "Built %s/%s | obs_dim=%d act_dim=%d horizon=%d shaped=%s",
            self.cfg.task,
            self.cfg.robot,
            obs_dim,
            self.action_space.shape[0],
            self.cfg.horizon,
            self.cfg.reward_shaping,
        )

    # ------------------------------------------------------------------ helpers

    def _flatten(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate the selected observation modalities into a flat float32 vector."""
        missing = [k for k in self._obs_keys if k not in obs_dict]
        if missing:
            raise KeyError(f"Missing obs keys {missing}. Available: {sorted(obs_dict)}")
        return np.concatenate([np.asarray(obs_dict[k]).ravel() for k in self._obs_keys]).astype(
            np.float32
        )

    def _is_success(self) -> bool:
        check = getattr(self._env, "_check_success", None)
        return bool(check()) if callable(check) else False

    # -------------------------------------------------------------- gym.Env API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # robosuite samples object placements from the *global* numpy RNG, so
            # per-instance seeding via self.np_random is not enough. This is a
            # robosuite limitation, not a Gymnasium one.
            np.random.seed(seed)
        self._elapsed_steps = 0
        return self._flatten(self._env.reset()), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high
        )
        obs_dict, reward, _robosuite_done, info = self._env.step(action)
        self._elapsed_steps += 1

        success = self._is_success()
        terminated = bool(success and self.cfg.terminate_on_success)
        truncated = bool(self._elapsed_steps >= self.cfg.horizon) and not terminated

        info = dict(info)
        info["is_success"] = success
        return self._flatten(obs_dict), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            self._env.render()
            return None
        if self.render_mode == "rgb_array":
            frame = self._env.sim.render(
                width=self.cfg.camera_width,
                height=self.cfg.camera_height,
                camera_name=self.cfg.camera_name,
            )
            return frame[::-1]  # MuJoCo returns bottom-up
        return None

    def close(self) -> None:
        self._env.close()


def make_lift_env(cfg: EnvConfig | None = None) -> gym.Env:
    """Factory used by SB3's ``make_vec_env`` and by the test-suite."""
    return RobosuiteLiftEnv(cfg)
