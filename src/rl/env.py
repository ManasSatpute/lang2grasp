"""Gymnasium-compatible wrapper around robosuite's ``Lift`` task (Franka Panda).

robosuite ships its own ``GymWrapper``, but it targets the legacy OpenAI Gym API;
Stable-Baselines3 speaks Gymnasium, so this wraps the raw robosuite environment
directly. Two details are handled explicitly for correctness: robosuite's ``done`` at
the horizon is reported as ``truncated``, not ``terminated`` (else SB3 wrongly
bootstraps a zero value there), and ``info["is_success"]`` is set every step so SB3's
``EvalCallback`` can report ``eval/success_rate``.

``"fingertip_force"`` is a genuine 6-dim MuJoCo sensor reading (``PandaGripperForce``,
see ``objects/force_gripper.py``), always part of the observation for every config
including the baseline. Whenever ``EnvConfig.object`` is set, the real contact force
is compared each step against the object's ``crush_force_N``: exceeding it applies a
reward penalty (``crush_penalty_coeff``) and by default ends the episode
(``terminate_on_crush``), unconditionally. Only force transmitted through a pad that
is *actually touching the object* counts (``_pads_touching_object``) -- the raw sensor
also reads table bumps and the pads closing on empty air, which are not crushes.
``grip_force_shaping`` separately adds an optional bonus for staying within
``grip_force_min_N``/``max_N``.

**Golden vs. extracted object.** ``EnvConfig.object`` is the real object actually
built into the MuJoCo scene. ``EnvConfig.extracted_object``, when set, is a separate
(possibly imperfect) ``ObjectParams`` -- e.g. an LLM's extraction -- that crush
behaviour, the grip-force bonus, and ``include_object_z``'s z-vector are computed from
instead, so the policy is trained against its *belief* about the object while the real
physics come from ``object``. Left unset, perception falls back to ``object``.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces

from objects import lift_object_task  # noqa: F401  -- registers "ParamLift" with robosuite
from objects.force_gripper import PAD_FORCE_SENSOR_NAMES  # noqa: F401 -- registers "PandaGripperForce"
from objects.object_params import (
    ObjectParams,
    Shape,
    object_params_to_noisy_z,
    sample_object_params,
)

LOGGER = logging.getLogger(__name__)

#: Observation key `_augment_obs_dict` adds when `EnvConfig.include_object_z` is set.
#: Not a robosuite-native obs_dict key, same as `fingertip_force` -- see
#: `DEFAULT_OBS_KEYS`'s docstring.
OBJECT_Z_KEY = "object_z"

#: robosuite's low-dim state keys, plus the genuine per-fingertip force sensor (see
#: module docstring and objects/force_gripper.py). `robot0_proprio-state` = joint
#: pos/vel, gripper, eef pose. `object-state` = cube pose + relative-to-eef vector.
#: `fingertip_force` = [left_fx,fy,fz, right_fx,fy,fz] (N), injected by this env's
#: reset()/step() -- not a robosuite-native obs_dict key.
DEFAULT_OBS_KEYS: tuple[str, ...] = ("robot0_proprio-state", "object-state", "fingertip_force")

#: `important_geoms` keys naming each finger's collision geoms, in `fingertip_force`
#: order (left = `[:3]`, right = `[3:]`). Each entry is tried in order, so a gripper
#: exposing only the coarser "left_finger"/"right_finger" grouping still works. Used by
#: `RobosuiteLiftEnv._pads_touching_object` to gate the crush check on real contact.
PAD_GEOM_KEYS: tuple[tuple[str, ...], ...] = (
    ("left_fingerpad", "left_finger"),
    ("right_fingerpad", "right_finger"),
)


@dataclass
class EnvConfig:
    """Everything needed to construct one :class:`RobosuiteLiftEnv`."""

    task: str = "Lift"
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    horizon: int = 500
    control_freq: int = 20
    #: robosuite's own default is sparse (False); shaped is needed for random
    #: exploration on a 7-DoF arm to ever see a learning signal.
    reward_shaping: bool = True
    #: End the episode the moment the cube is lifted. Off by default so episode
    #: return stays comparable across runs.
    terminate_on_success: bool = False
    obs_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_OBS_KEYS)
    render_mode: str | None = None
    camera_name: str = "agentview"
    camera_height: int = 256
    camera_width: int = 256
    #: The "golden"/ground-truth object actually built into the MuJoCo scene
    #: (`ParamLift`). None reproduces the stock robosuite Lift cube. Mutually
    #: exclusive with `randomize_object`.
    object: ObjectParams | None = None
    #: A separate, possibly-imperfect `ObjectParams` (e.g. an LLM extraction) that
    #: crush behaviour, the grip-force bonus, and `include_object_z` are computed
    #: from instead of `object`. None falls back to `object`. See module docstring.
    extracted_object: ObjectParams | None = None

    #: Paradigm-switch mode: `ParamLift` draws a fresh `ObjectParams` every episode
    #: instead of one fixed per-object specialist. Forces `hard_reset=True`, which is
    #: markedly slower per reset (a shape/size change needs a MuJoCo recompile).
    randomize_object: bool = False
    #: Only meaningful when `randomize_object=True`; restricts sampling to this
    #: subset of shapes, or all of them if None.
    randomize_shapes: tuple[Shape, ...] | None = None

    #: Append a (possibly noisy) object parameter vector `z` to the observation --
    #: the `pi_param` paradigm-switch variant. Requires `object` or `randomize_object`.
    include_object_z: bool = False
    #: Relative Gaussian noise on z's continuous dims; 0.0 = exact/noiseless z.
    object_z_noise_std: float = 0.1

    #: `pi_blind+hist`'s memory: when > 0, wraps observations (`HistoryObsWrapper`)
    #: with a rolling window of the last `history_len` steps' proprioception + force.
    #: 0 = disabled.
    history_len: int = 0

    #: Reward bonus each step the real fingertip force stays within
    #: [grip_force_min_N, grip_force_max_N]. Requires `object`. Off by default, and
    #: does not gate crush behaviour (below) -- fragility isn't optional.
    grip_force_shaping: bool = False
    #: Bonus applied when grip_force_shaping is True and the force is in-window.
    grip_force_bonus: float = 0.1
    #: Per-Newton reward penalty applied whenever `object` is set and the real
    #: fingertip force exceeds `object.crush_force_N`. Unconditional.
    crush_penalty_coeff: float = 0.1
    #: End the episode the step fingertip force first exceeds crush_force_N.
    #: Unconditional, same as crush_penalty_coeff.
    terminate_on_crush: bool = True

    def __post_init__(self) -> None:
        # JSON round-trips tuples back as lists; normalise so a config and its
        # reloaded snapshot compare equal.
        self.obs_keys = tuple(self.obs_keys)
        if self.randomize_shapes is not None:
            self.randomize_shapes = tuple(self.randomize_shapes)
        if isinstance(self.object, dict):
            self.object = ObjectParams(**self.object)
        if isinstance(self.extracted_object, dict):
            self.extracted_object = ObjectParams(**self.extracted_object)
        if self.randomize_object and self.object is not None:
            raise ValueError("randomize_object=True is mutually exclusive with a fixed `object`.")
        has_object = self.object is not None or self.randomize_object
        if self.grip_force_shaping and not has_object:
            raise ValueError("grip_force_shaping requires `object` or randomize_object.")
        if self.include_object_z and not has_object:
            raise ValueError("include_object_z requires `object` or randomize_object.")
        if self.extracted_object is not None and self.object is None:
            raise ValueError(
                "extracted_object requires a fixed `object` (it's a separate, possibly "
                "imperfect ObjectParams for the *same* golden object; meaningless "
                "without one -- and not currently supported alongside randomize_object)."
            )


def resolve_controller_config(controller_name: str, robot: str) -> dict[str, Any]:
    """Return a controller config across the robosuite 1.4 / 1.5 API split.

    robosuite 1.5 replaced the flat ``load_controller_config(default_controller=...)``
    dict with a composite (per-body-part) config, looked up by a different name
    convention. This loads the named part-controller block and lets robosuite's own
    ``refactor_composite_controller_config`` wrap it into the composite shape, so any
    ``controller_name`` (not just the default) resolves correctly on either version.
    """
    try:
        from robosuite.controllers import load_part_controller_config
        from robosuite.controllers.composite.composite_controller_factory import (
            is_part_controller_config,
            refactor_composite_controller_config,
        )
    except ImportError:  # robosuite < 1.5
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller=controller_name)

    part_cfg = load_part_controller_config(default_controller=controller_name)
    assert is_part_controller_config(part_cfg), (
        f"{controller_name!r} did not resolve to a part controller config: {part_cfg!r}"
    )
    # Single-arm robots (Panda included) are keyed "right" in robosuite's composite
    # body_parts dict -- this project's env has no bimanual/mobile-base support.
    composite = refactor_composite_controller_config(part_cfg, robot, ["right"])
    assert composite["body_parts"]["right"]["type"] == part_cfg["type"], (
        f"expected {controller_name!r} to land in body_parts['right'], got "
        f"{composite['body_parts']['right'].get('type')!r}"
    )
    return composite


class RobosuiteLiftEnv(gym.Env):
    """A single-agent, state-observation Gymnasium view of a robosuite task."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self._obs_keys = tuple(self.cfg.obs_keys)
        if self.cfg.include_object_z and OBJECT_Z_KEY not in self._obs_keys:
            self._obs_keys = (*self._obs_keys, OBJECT_Z_KEY)
        self.render_mode = self.cfg.render_mode
        self._current_object: ObjectParams | None = None  # set in reset(); see _resolve_current_object

        object_kwargs: dict[str, Any] = {}
        env_name = self.cfg.task
        hard_reset = False
        if self.cfg.object is not None:
            if self.cfg.task != "Lift":
                raise ValueError(
                    f"cfg.object is only supported for task='Lift', got {self.cfg.task!r}"
                )
            env_name = "ParamLift"
            object_kwargs["object_params"] = self.cfg.object
        elif self.cfg.randomize_object:
            if self.cfg.task != "Lift":
                raise ValueError(
                    f"cfg.randomize_object is only supported for task='Lift', got {self.cfg.task!r}"
                )
            env_name = "ParamLift"
            shapes = self.cfg.randomize_shapes
            object_kwargs["object_params"] = (
                (lambda: sample_object_params(shapes=shapes)) if shapes else sample_object_params
            )
            # A shape/size change needs a MuJoCo recompile every reset.
            hard_reset = True

        # PandaGripperForce's fingertip sensor is Panda-specific; fall back to the
        # robot's own default gripper for anything else (degrades to an all-zero
        # fingertip_force instead of crashing).
        if self.cfg.robot == "Panda":
            object_kwargs["gripper_types"] = "PandaGripperForce"
        else:
            LOGGER.warning(
                "robot=%r has no fingertip force sensor (PandaGripperForce is "
                "Panda-only); 'fingertip_force' in the observation will be all-zero.",
                self.cfg.robot,
            )

        self.resolved_controller_config = resolve_controller_config(self.cfg.controller, self.cfg.robot)
        LOGGER.info(
            "Resolved controller %r for %s -> %s",
            self.cfg.controller,
            self.cfg.robot,
            self.resolved_controller_config,
        )

        self._env = suite.make(
            env_name=env_name,
            robots=self.cfg.robot,
            controller_configs=self.resolved_controller_config,
            has_renderer=self.render_mode == "human",
            has_offscreen_renderer=self.render_mode == "rgb_array",
            use_camera_obs=False,  # state-only: keeps the MLP policy small and fast
            use_object_obs=True,
            reward_shaping=self.cfg.reward_shaping,
            horizon=self.cfg.horizon,
            control_freq=self.cfg.control_freq,
            ignore_done=True,  # this wrapper owns episode termination; see module docstring
            hard_reset=hard_reset,
            **object_kwargs,
        )

        low, high = self._env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

        self._fingertip_force_warned = False  # log the degradation warning once, not every step
        self._pad_geoms_warned = False
        raw_obs_dict = self._env.reset()
        self._current_object = self._resolve_current_object()
        obs_dim = self._flatten(self._augment_obs_dict(raw_obs_dict)).shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._elapsed_steps = 0
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

    def obs_slices(self) -> dict[str, slice]:
        """The contiguous flat-observation slice for each key in `_obs_keys`. Requires
        `reset()` to have been called first. Used by `HistoryObsWrapper`."""
        if not self.last_obs_dict:
            raise RuntimeError("obs_slices() requires reset() to have been called first.")
        slices: dict[str, slice] = {}
        offset = 0
        for key in self._obs_keys:
            dim = int(np.asarray(self.last_obs_dict[key]).size)
            slices[key] = slice(offset, offset + dim)
            offset += dim
        return slices

    def _is_success(self) -> bool:
        check = getattr(self._env, "_check_success", None)
        return bool(check()) if callable(check) else False

    def _read_fingertip_forces_N(self) -> np.ndarray:
        """Real per-pad 3-axis contact force (N), (6,) float32. All-zero (with a
        one-time warning) if the active gripper has no force sensors."""
        robot = self._env.robots[0]
        arm = robot.arms[0]
        gripper = robot.gripper[arm]
        sensors = gripper.important_sensors
        if not all(name in sensors for name in PAD_FORCE_SENSOR_NAMES):
            if not self._fingertip_force_warned:
                LOGGER.warning(
                    "Active gripper %s has no %s sensors; 'fingertip_force' will be "
                    "all-zero for this env.",
                    type(gripper).__name__,
                    PAD_FORCE_SENSOR_NAMES,
                )
                self._fingertip_force_warned = True
            return np.zeros(6, dtype=np.float32)

        forces = [robot.get_sensor_measurement(sensors[name]) for name in PAD_FORCE_SENSOR_NAMES]
        return np.concatenate(forces).astype(np.float32)

    def _pads_touching_object(self) -> tuple[bool, bool]:
        """``(left, right)``: is each finger pad actually in contact with the object?

        The fingertip sensor reads whatever load is transmitted through that pad --
        the pad bumping the table, the two pads meeting on empty air, or just the arm
        accelerating -- none of which is the gripper squeezing the object. Without this
        gate, a `crush_force_N` in the 5-12 N range (every fragile object, and the whole
        `width_mass_set`) terminates episodes during the reach phase, so training never
        reaches a grasp and `eval/success_rate` stays pinned at 0. Falls back to
        ``(True, True)`` -- i.e. the old ungated reading -- if the gripper doesn't
        expose per-finger geoms to check against.
        """
        obj = getattr(self._env, "cube", None)
        if obj is None:
            return (True, True)

        robot = self._env.robots[0]
        gripper = robot.gripper[robot.arms[0]]
        geoms = gripper.important_geoms
        touching: list[bool] = []
        for keys in PAD_GEOM_KEYS:
            pad_geoms = next((geoms[key] for key in keys if key in geoms), None)
            if pad_geoms is None:
                if not self._pad_geoms_warned:
                    LOGGER.warning(
                        "Gripper %s exposes none of %s in important_geoms; the crush "
                        "check falls back to the ungated fingertip force.",
                        type(gripper).__name__,
                        keys,
                    )
                    self._pad_geoms_warned = True
                return (True, True)
            touching.append(bool(self._env.check_contact(pad_geoms, obj)))
        return touching[0], touching[1]

    def _augment_obs_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """A copy of robosuite's obs_dict with the real fingertip-force reading (and,
        if `include_object_z`, this episode's noisy object-parameter vector) added."""
        obs_dict = dict(obs_dict)
        obs_dict["fingertip_force"] = self._read_fingertip_forces_N()
        if self.cfg.include_object_z:
            obs_dict[OBJECT_Z_KEY] = object_params_to_noisy_z(
                self._current_object, self.cfg.object_z_noise_std
            )
        return obs_dict

    def _resolve_current_object(self) -> ObjectParams | None:
        """This episode's *perceived* `ObjectParams` (see module docstring, "Golden vs.
        extracted object"). Must be called after `self._env.reset()`."""
        if self.cfg.object is not None:
            return self.cfg.extracted_object or self.cfg.object
        if self.cfg.randomize_object:
            return self._env.object_params
        return None

    def _grip_force_bonus_term(self, force_N: float) -> float:
        """Bonus for holding within the object's safe window. Crush is handled in step()."""
        obj = self._current_object
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
            # robosuite samples object placement (and, under randomize_object, the
            # episode's ObjectParams) from the global numpy RNG, not a per-instance one.
            np.random.seed(seed)
        self._elapsed_steps = 0
        raw_obs_dict = self._env.reset()
        self._current_object = self._resolve_current_object()
        obs_dict = self._augment_obs_dict(raw_obs_dict)
        self.last_obs_dict = obs_dict
        return self._flatten(obs_dict), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64), self.action_space.low, self.action_space.high
        )
        obs_dict, reward, _robosuite_done, info = self._env.step(action)
        obs_dict = self._augment_obs_dict(obs_dict)
        self.last_obs_dict = obs_dict
        self._elapsed_steps += 1
        reward = float(reward)

        info = dict(info)
        crushed = False
        if self._current_object is not None:
            fingertip_force = obs_dict["fingertip_force"]
            left_N = float(np.linalg.norm(fingertip_force[:3]))
            right_N = float(np.linalg.norm(fingertip_force[3:]))
            # Only load transmitted through a pad that is genuinely on the object counts
            # as grip force -- see _pads_touching_object for why this gate is load-bearing.
            left_touching, right_touching = self._pads_touching_object()
            contact_force_N = max(
                left_N if left_touching else 0.0, right_N if right_touching else 0.0
            )
            info["fingertip_force_N"] = contact_force_N
            #: Ungated reading, for diagnosing the gate itself. Not used by the reward.
            info["fingertip_force_raw_N"] = max(left_N, right_N)

            crush_excess_N = max(0.0, contact_force_N - self._current_object.crush_force_N)
            if crush_excess_N > 0.0:
                reward -= self.cfg.crush_penalty_coeff * crush_excess_N
                crushed = self.cfg.terminate_on_crush

            if self.cfg.grip_force_shaping:
                reward += self._grip_force_bonus_term(contact_force_N)

        info["crushed"] = crushed
        success = self._is_success()
        terminated = bool(crushed or (success and self.cfg.terminate_on_success))
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


#: Modalities `HistoryObsWrapper` historizes by default -- proprioception + fingertip
#: force. Not object-state (cube pose), which is fully informative each single step.
DEFAULT_HISTORY_KEYS: tuple[str, ...] = ("robot0_proprio-state", "fingertip_force")


class HistoryObsWrapper(gym.Wrapper):
    """Appends a rolling window of the last ``history_len`` steps' proprioception +
    fingertip force to the observation -- ``pi_blind+hist``'s only source of implicit
    system identification. Output layout: ``[current obs, history block]``, oldest
    step first, zero-padded at episode start. Pairs with
    ``rl.policies.HistoryGRUExtractor``, which splits this back apart.
    """

    def __init__(
        self,
        env: RobosuiteLiftEnv,
        history_len: int,
        history_keys: tuple[str, ...] = DEFAULT_HISTORY_KEYS,
    ) -> None:
        super().__init__(env)
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        self.history_len = history_len
        self.history_keys = history_keys

        obs, _ = env.reset()  # populates env.last_obs_dict so obs_slices() can size the history block
        slices = env.obs_slices()
        self._step_slices = [slices[k] for k in history_keys]
        self.step_dim = sum(s.stop - s.start for s in self._step_slices)
        self._current_dim = obs.shape[0]
        self._history: deque[np.ndarray] = deque(maxlen=history_len)

        flat_dim = self._current_dim + history_len * self.step_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def _step_vec(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs[s] for s in self._step_slices]).astype(np.float32)

    def _stacked_obs(self, obs: np.ndarray) -> np.ndarray:
        self._history.append(self._step_vec(obs))
        while len(self._history) < self.history_len:  # only on the first call after reset()
            self._history.appendleft(np.zeros(self.step_dim, dtype=np.float32))
        history_block = np.concatenate(list(self._history))
        return np.concatenate([obs, history_block]).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._history.clear()
        return self._stacked_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._stacked_obs(obs), reward, terminated, truncated, info


def make_lift_env(cfg: EnvConfig | None = None) -> gym.Env:
    """Factory used by SB3's ``make_vec_env`` and by the test-suite."""
    cfg = cfg or EnvConfig()
    env: gym.Env = RobosuiteLiftEnv(cfg)
    if cfg.history_len > 0:
        env = HistoryObsWrapper(env, history_len=cfg.history_len)
    return env


def history_dims(cfg: EnvConfig, history_keys: tuple[str, ...] = DEFAULT_HISTORY_KEYS) -> tuple[int, int, int]:
    """``(current_dim, history_len, step_dim)`` for ``rl.policies.HistoryGRUExtractor``,
    measured from a throwaway env so callers never hardcode them. Requires
    ``cfg.history_len > 0``."""
    if cfg.history_len <= 0:
        raise ValueError(f"cfg.history_len must be positive, got {cfg.history_len}")
    base_env = RobosuiteLiftEnv(cfg)
    try:
        obs, _ = base_env.reset()
        current_dim = obs.shape[0]
        slices = base_env.obs_slices()
        step_dim = sum(slices[k].stop - slices[k].start for k in history_keys)
    finally:
        base_env.close()
    return current_dim, cfg.history_len, step_dim
