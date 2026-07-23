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

Optional grip-force-aware reward shaping (``EnvConfig.grip_force_shaping``) uses
``ObjectParams.grip_force_min_N``/``grip_force_max_N``/``crush_force_N``/``spring_Npm``,
so an LLM-described object's force window actually influences this training loop too,
not just the object's geometry/density/friction. Per-finger contact force is
*estimated* from the Panda gripper's finger joint positions
(``obs_dict["robot0_gripper_qpos"]``, always present for any gripper-equipped robot --
see ``robosuite.robots.robot.Robot._create_arm_sensors``) via
``ObjectParams.reaction_force_N``'s spring-compression model. It is an estimate, not a
true contact-force sensor reading, and assumes a two-finger parallel gripper whose
joint qpos sum to aperture the way Panda's does (confirmed against robosuite 1.5's
``panda_gripper.xml``: finger joints range [0, 0.04] and [-0.04, 0] metres, so
aperture = qpos[0] - qpos[1] spans the Panda's ~80mm opening). A different gripper
model would need this remapped.
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

    #: Add a grip-force-aware term to the reward each step, using `object`'s
    #: grip_force_min_N/max_N/crush_force_N (see module docstring). Requires `object`
    #: to be set. Off by default -- existing configs are byte-for-byte unaffected.
    grip_force_shaping: bool = False
    #: Reward added each step the estimated per-finger contact force falls inside
    #: [object.grip_force_min_N, object.grip_force_max_N] -- a secure, non-crushing hold.
    grip_force_bonus: float = 0.1
    #: Reward subtracted each step the estimated per-finger contact force exceeds
    #: object.crush_force_N. Deliberately larger than grip_force_bonus: crushing a
    #: fragile object should outweigh several steps of a good hold.
    crush_penalty: float = 1.0

    def __post_init__(self) -> None:
        # JSON round-trips a tuple back as a list. Normalise on construction so a
        # config and its reloaded snapshot compare equal, and so `rollout.py`
        # rebuilds the exact observation layout the policy was trained on.
        self.obs_keys = tuple(self.obs_keys)
        # Same round-trip concern: a loaded JSON snapshot hands back a plain dict.
        if isinstance(self.object, dict):
            self.object = ObjectParams(**self.object)
        if self.grip_force_shaping and self.object is None:
            raise ValueError("grip_force_shaping requires `object` to be set.")


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
        self._grip_force_warned = False  # log the graceful-degradation warning once, not every step
        self.last_obs_dict: dict[str, np.ndarray] = {}  # raw robosuite obs, e.g. for rl.sparse_seed
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

    def _estimate_grip_force_N(self, obs_dict: dict[str, np.ndarray]) -> float | None:
        """Estimate per-finger contact force (N) from gripper aperture.

        Returns None (and warns once) if the expected observable isn't there -- e.g. a
        gripper with a different joint convention than the two-finger Panda this was
        written against. See module docstring for the aperture formula's derivation.
        """
        qpos = obs_dict.get("robot0_gripper_qpos")
        if qpos is None or np.asarray(qpos).shape != (2,):
            if not self._grip_force_warned:
                LOGGER.warning(
                    "grip_force_shaping: expected obs_dict['robot0_gripper_qpos'] with "
                    "shape (2,), got %s. Disabling force-based reward shaping for this env.",
                    None if qpos is None else np.asarray(qpos).shape,
                )
                self._grip_force_warned = True
            return None

        aperture_mm = (float(qpos[0]) - float(qpos[1])) * 1000.0
        return self.cfg.object.reaction_force_N(aperture_mm)

    def _grip_force_reward_term(self, force_N: float) -> float:
        """Bonus for holding within the object's safe window, penalty for exceeding it."""
        obj = self.cfg.object
        if force_N > obj.crush_force_N:
            return -self.cfg.crush_penalty
        if obj.grip_force_min_N <= force_N <= obj.grip_force_max_N:
            return self.cfg.grip_force_bonus
        return 0.0

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
        obs_dict = self._env.reset()
        self.last_obs_dict = obs_dict
        return self._flatten(obs_dict), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high
        )
        obs_dict, reward, _robosuite_done, info = self._env.step(action)
        self.last_obs_dict = obs_dict
        self._elapsed_steps += 1
        reward = float(reward)

        info = dict(info)
        if self.cfg.grip_force_shaping:
            force_N = self._estimate_grip_force_N(obs_dict)
            if force_N is not None:
                reward += self._grip_force_reward_term(force_N)
                info["grip_force_N"] = force_N

        success = self._is_success()
        terminated = bool(success and self.cfg.terminate_on_success)
        truncated = bool(self._elapsed_steps >= self.cfg.horizon) and not terminated

        info["is_success"] = success
        return self._flatten(obs_dict), reward, terminated, truncated, info

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
