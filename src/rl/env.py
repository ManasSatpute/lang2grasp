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

Fingertip force is a genuine MuJoCo sensor reading, not an estimate: every env uses
``PandaGripperForce`` (``objects/force_gripper.py``), a Panda gripper variant with a
3-axis ``<force>`` sensor on each fingertip pad, and ``"fingertip_force"`` -- a 6-dim
``[left_fx,fy,fz, right_fx,fy,fz]`` vector -- is always part of the observation
(``DEFAULT_OBS_KEYS``), for every config, baseline included. That matters: a real
robot has this sensor regardless of whether it's holding an LLM-described object, so
a force-conditioned policy's edge over the generic baseline can't just be "I have a
sensor you don't."

Crush behaviour is object-specific (keyed off ``ObjectParams.crush_force_N``, which
only exists once ``EnvConfig.object`` is set): whenever ``object`` is set, every step
computes ``contact_force_N = max(|left force|, |right force|)`` from the real sensor,
and if it exceeds the *perceived* object's ``crush_force_N`` the episode pays a reward
penalty proportional to the excess (``EnvConfig.crush_penalty_coeff``) and, by default,
terminates (``EnvConfig.terminate_on_crush``) -- this is unconditional (not gated by
``grip_force_shaping``), since fragility shouldn't be an opt-in experiment. Separately,
``EnvConfig.grip_force_shaping`` still exists, but now controls *only* the optional
"safe-hold" bonus for staying within the perceived object's ``grip_force_min_N``/``max_N``.

**Golden physics vs. extracted perception.** ``EnvConfig.object`` is what actually gets
built into the MuJoCo scene (``ParamLift``) -- the real/"golden" object, typically loaded
from a trusted source (e.g. ``extraction.param_prompts.golden_object_params``) rather
than an LLM guess. ``EnvConfig.extracted_object``, when set, is a *separate* (possibly
imperfect) ``ObjectParams`` -- e.g. an LLM's extraction from a text prompt -- that the
crush penalty/termination, the grip-force bonus, and ``include_object_z``'s z-vector are
computed from instead: the policy is trained against what it *believes* about the
object (from extraction), while what actually gets lifted (and how much it actually
masses/how it actually slides) is the golden object. This is what lets a per-object run
test robustness to extraction error, rather than extraction error simply not existing
(golden == extracted) as it did before. ``extracted_object=None`` (the default) falls
back to using ``object`` for perception too -- byte-for-byte the old behaviour.
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
    #: The object actually built into the MuJoCo scene (`ParamLift`) -- the "golden"/
    #: ground-truth physical object. None reproduces the stock robosuite Lift cube
    #: exactly (env_name="Lift", no ParamLift involved at all), unless
    #: `randomize_object` is set instead (see below). Mutually exclusive with
    #: `randomize_object`: there's no single fixed object once every episode draws
    #: its own.
    object: ObjectParams | None = None
    #: A separate, possibly-imperfect `ObjectParams` (e.g. an LLM's extraction from a
    #: text prompt) that crush penalty/termination, the grip-force bonus, and
    #: `include_object_z`'s z-vector are computed from *instead of* `object` -- i.e.
    #: what the policy is rewarded/conditioned on is its belief about the object, not
    #: the real physics it's actually lifting (`object`). None (the default) falls
    #: back to `object` for perception too -- the old, coupled behaviour. Only
    #: meaningful when `object` is set. See module docstring, "Golden physics vs.
    #: extracted perception".
    extracted_object: ObjectParams | None = None

    #: The paradigm-switch mode (see `objects.object_params.sample_object_params`):
    #: instead of one fixed per-object specialist, `ParamLift` draws a fresh
    #: continuously-distributed `ObjectParams` (shape/size/density/friction) every
    #: episode. Forces `hard_reset=True` in `suite.make` (a shape/size change needs a
    #: MuJoCo recompile, unlike the fixed-object path's reused compiled model), which
    #: is markedly slower per reset -- see `RobosuiteLiftEnv.__init__`.
    randomize_object: bool = False
    #: Only meaningful when `randomize_object=True`. Restricts sampling to this subset
    #: of shapes; None samples uniformly over all of `sample_object_params`'s default
    #: shapes.
    randomize_shapes: tuple[Shape, ...] | None = None

    #: Append a (possibly noisy) object physical-parameter vector `z` to the
    #: observation -- `objects.object_params.object_params_to_noisy_z`, keyed
    #: `rl.env.OBJECT_Z_KEY`. This is the "informed" pi_param variant that
    #: FiLM-conditions on an estimated z, as opposed to pi_blind/pi_blind+hist, which
    #: never see z at all. Requires `object` or `randomize_object`.
    include_object_z: bool = False
    #: Relative (fractional) Gaussian noise applied to z's continuous dims (size,
    #: density, friction) -- z models a noisy sysID-style estimate, not ground truth.
    #: 0.0 = exact/noiseless z. Only read when `include_object_z=True`.
    object_z_noise_std: float = 0.1

    #: pi_blind+hist's memory: when > 0, observations are wrapped
    #: (`rl.env.HistoryObsWrapper`) to append a rolling window of the last
    #: `history_len` steps' proprioception + fingertip force, oldest-first,
    #: zero-padded at episode start. Paired with `rl.policies.HistoryGRUExtractor`,
    #: which knows how to split the wrapped observation back apart. 0 = disabled
    #: (pi_blind / pi_param / the per-object specialists all leave this at 0).
    history_len: int = 0

    #: Add a reward bonus each step the real fingertip force falls inside
    #: [object.grip_force_min_N, object.grip_force_max_N] -- a secure, non-crushing
    #: hold. Requires `object` to be set. Off by default -- existing configs are
    #: byte-for-byte unaffected. Does NOT gate crush behaviour any more (see
    #: crush_penalty_coeff/terminate_on_crush below and the module docstring) --
    #: fragility isn't optional the way this bonus is.
    grip_force_shaping: bool = False
    #: Reward added each step the real per-finger contact force falls inside the
    #: window above. Only applied when grip_force_shaping is True.
    grip_force_bonus: float = 0.1
    #: Per-Newton reward penalty (lambda), applied whenever `object` is set and the
    #: real fingertip force exceeds object.crush_force_N: reward -=
    #: crush_penalty_coeff * (contact_force_N - object.crush_force_N). Unconditional
    #: on `object` being set -- not gated by grip_force_shaping.
    crush_penalty_coeff: float = 0.1
    #: End the episode the step fingertip force first exceeds object.crush_force_N.
    #: Unconditional on `object` being set, same as crush_penalty_coeff -- a crushed
    #: object is a real terminal state, not an optional shaping choice.
    terminate_on_crush: bool = True

    def __post_init__(self) -> None:
        # JSON round-trips a tuple back as a list. Normalise on construction so a
        # config and its reloaded snapshot compare equal, and so `rollout.py`
        # rebuilds the exact observation layout the policy was trained on.
        self.obs_keys = tuple(self.obs_keys)
        if self.randomize_shapes is not None:
            self.randomize_shapes = tuple(self.randomize_shapes)
        # Same round-trip concern: a loaded JSON snapshot hands back a plain dict.
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
    dict with a composite (per-body-part) config. Composite configs are looked up by
    *composite* name (e.g. ``"BASIC"``) in ``REGISTERED_COMPOSITE_CONTROLLERS_DICT`` --
    ``"OSC_POSE"``/``"OSC_POSITION"``/``"JOINT_VELOCITY"``/etc. are *part* (single-arm)
    controller names, and asserting one of those against that registry crashes.

    A hardcoded ``"BASIC"`` here would have silently discarded ``controller_name``
    (it happens to match "BASIC" for the OSC_POSE default this project ships, but
    would keep silently landing there for any other value in a config's ``controller``
    field). Instead, use robosuite's own upgrade path: load the named part-controller
    block, then let ``refactor_composite_controller_config`` wrap it into the composite
    shape -- confirmed against the installed robosuite 1.5.1 source
    (``robosuite/controllers/composite/composite_controller_factory.py``).
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
            # A shape/size change needs a MuJoCo recompile, so a fresh per-episode
            # sample requires hard_reset=True -- ~10x slower per reset than the fixed
            # per-object path below (see that branch's `hard_reset=False` comment).
            hard_reset = True

        # The genuine fingertip force sensor (module docstring) is Panda-specific --
        # PandaGripperForce's derived XML is built from panda_gripper.xml. Every
        # config this project ships uses robot="Panda" (see README.md); fall back to
        # that robot's own default gripper for anything else rather than erroring, so
        # an exploratory non-Panda run degrades to "no fingertip_force sensor" instead
        # of crashing.
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
            ignore_done=True,  # we own episode termination; see module docstring
            # False (fixed object / stock cube): ~10x faster resets, re-uses the
            # compiled MjModel. True (randomize_object): forced above, since a fresh
            # per-episode shape/size sample needs a MuJoCo recompile every reset.
            hard_reset=hard_reset,
            **object_kwargs,
        )

        low, high = self._env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

        self._fingertip_force_warned = False  # log the degradation warning once, not every step
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
        """The contiguous flat-observation slice occupied by each key in `_obs_keys`,
        in order. Requires `reset()` to have been called at least once (sizes each
        modality off `self.last_obs_dict`). Used by `HistoryObsWrapper` to pull out
        the proprio/force sub-vectors it historizes, without hardcoding `_flatten`'s
        layout a second time.
        """
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
        """Real per-pad 3-axis contact force (N) from PandaGripperForce's MuJoCo sensors.

        Returns a (6,) float32 vector ``[left_fx,fy,fz, right_fx,fy,fz]``. All-zero
        (with a one-time warning) if the active gripper doesn't have these sensors --
        e.g. cfg.robot != "Panda", see __init__. See module docstring and
        objects/force_gripper.py for how the sensors are added.
        """
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
        """This episode's *perceived* `ObjectParams` -- what crush penalty/
        termination, the grip-force bonus, and `include_object_z`'s z-vector are
        computed from. Fixed-object runs use `cfg.extracted_object` when set
        (falling back to `cfg.object`, the golden physical object, otherwise -- see
        module docstring, "Golden physics vs. extracted perception"); randomized runs
        use whatever `ParamLift` actually sampled this episode (there is no separate
        extracted-vs-golden split under `randomize_object` yet); no `object` at all
        (the stock-cube baseline) has no perceived object either.

        Must be called *after* `self._env.reset()`: in the `randomize_object` case,
        `ParamLift._load_model()` (invoked by that reset, since `hard_reset=True`)
        has already drawn this episode's sample by the time `reset()` returns, and
        `self._env.object_params` reflects it.
        """
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
            # robosuite samples object placements (and, when randomize_object is set,
            # this episode's ObjectParams -- see sample_object_params) from the
            # *global* numpy RNG, so per-instance seeding via self.np_random is not
            # enough. This is a robosuite limitation, not a Gymnasium one.
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
            contact_force_N = float(
                max(np.linalg.norm(fingertip_force[:3]), np.linalg.norm(fingertip_force[3:]))
            )
            info["fingertip_force_N"] = contact_force_N

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


#: Modalities `HistoryObsWrapper` historizes by default -- proprioception + the real
#: fingertip force sensor. Not object-state (cube pose): that's fully informative each
#: single step, so there's nothing for a temporal window to add there. See
#: `EnvConfig.history_len`'s docstring.
DEFAULT_HISTORY_KEYS: tuple[str, ...] = ("robot0_proprio-state", "fingertip_force")


class HistoryObsWrapper(gym.Wrapper):
    """Appends a rolling window of the last ``history_len`` steps' proprioception +
    fingertip force to the observation -- pi_blind+hist's only source of implicit
    system identification, since it (like pi_blind) never sees an object-parameter z.

    Output layout: ``[current full obs (unchanged, dim D), history block
    (history_len * step_dim, oldest-first, zero-padded at episode start)]``.
    ``rl.policies.HistoryGRUExtractor`` is the matching SB3 features extractor that
    knows how to split this back apart into ``(D, history_len, step_dim)`` and run a
    GRU over the history block -- the two must agree on ``history_len``/``history_keys``.
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
    """``(current_dim, history_len, step_dim)`` for ``rl.policies.HistoryGRUExtractor``.

    Builds (and immediately closes) one throwaway, un-wrapped env to measure the
    dimensions ``HistoryObsWrapper``/``HistoryGRUExtractor`` need to agree on, so a
    caller (``scripts/train_paradigm.py``) never has to hardcode them by hand.
    Requires ``cfg.history_len > 0``.
    """
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
