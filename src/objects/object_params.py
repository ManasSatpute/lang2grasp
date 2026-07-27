"""Physical + grasp-relevant parameters for an LLM-described object.

Two groups of fields:

- **Simulation fields** (``shape``, ``size``, ``density``, ``friction``) map directly
  onto robosuite's primitive objects (``BoxObject``/``CylinderObject``/``BallObject``,
  see ``lift_object_task.build_mujoco_object``), which take exactly this shape/size/
  density/friction quartet. ``friction`` is robosuite's own 3-tuple: (sliding,
  torsional, rolling).
- **Grasp-descriptive fields** (``mass_class``, ``fragile``, ``grip_force_min_N``,
  ``grip_force_max_N``, ``spring_Npm``, ``crush_force_N``) are metadata carried
  through the pipeline (extraction -> training config -> rollout output).
  ``grip_force_min_N``/``grip_force_max_N``/``crush_force_N`` drive `rl/env.py`'s
  reward shaping, keyed off a **genuine MuJoCo fingertip force sensor**
  (``objects/force_gripper.py``), not an estimate: crush penalty/termination
  (``crush_force_N``) are unconditional whenever ``EnvConfig.object`` is set, and the
  optional safe-hold bonus (``EnvConfig.grip_force_shaping``, off by default) compares
  the same real reading against ``grip_force_min_N``/``max_N``. Unlike
  ``grip_force_max_N`` (a friendlier training-time target range), ``crush_force_N`` is
  the harder physical damage threshold; the two can and do differ in scale for the
  same object. ``mass_class`` and ``fragile`` remain purely descriptive -- they don't
  change the physics (``fragile`` objects already get a low ``crush_force_N`` instead)
  or the reward directly. ``reaction_force_N``/``spring_Npm`` below predate the real
  sensor and are no longer read by `rl/env.py`; kept as a stiffness descriptor still
  carried through extraction, not currently consumed by training.

Values are clamped to ranges that stay graspable by a Panda parallel-jaw gripper and
numerically stable in MuJoCo. An LLM extrapolating from a text prompt occasionally
returns something wild (a "boulder" at 50 kg, a friction of 9) -- clamping here, once,
means every caller (env, tests, rollout) sees only safe values.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

LOGGER = logging.getLogger(__name__)

Shape = Literal["box", "cylinder", "ball"]
MassClass = Literal["light", "medium", "heavy"]

#: Per-shape, per-component (lo, hi) bounds in metres. The gripper grasps across
#: a diameter, not along an object's length, so a cylinder's radius is bounded
#: like a graspable width (~1-6cm) while its half-height is allowed to run
#: longer (e.g. a bottle or brick) without affecting whether the Panda's
#: parallel-jaw fingers (~8cm max aperture) can close on it.
_SIZE_BOUNDS_M: dict[str, tuple[tuple[float, float], ...]] = {
    "box": ((0.01, 0.06), (0.01, 0.06), (0.01, 0.06)),
    "cylinder": ((0.005, 0.06), (0.01, 0.15)),
    "ball": ((0.01, 0.06),),
}
_SIZE_DIMS: dict[Shape, int] = {"box": 3, "cylinder": 2, "ball": 1}

#: kg/m^3. Wide range to span thin-walled glass/hollow objects (~300) through
#: solid steel (~7850) without letting an LLM return a nonsensical outlier.
_DENSITY_RANGE = (50.0, 9000.0)
#: (sliding, torsional, rolling) coefficient-of-friction bounds. MuJoCo's own
#: defaults are (1.0, 0.005, 0.0001); real materials span roughly 0.05 (slick)
#: to 1.5 (rubbery) on sliding friction, with torsional/rolling much smaller.
_FRICTION_BOUNDS: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (0.05, 1.5),
    (0.0, 0.05),
    (0.0, 0.01),
)
#: Newtons, per finger. Below ~0.1N nothing meaningfully resists gravity; above
#: ~200N is well past what a Panda parallel-jaw gripper can exert.
_GRIP_FORCE_RANGE = (0.1, 200.0)
#: N/m. 20 ~ a very soft/compliant object, 10000 ~ a rigid one (brick-like).
_SPRING_RANGE = (20.0, 10000.0)
#: Newtons, per finger -- the literal contact force that damages the object.
#: Deliberately a much wider range than _GRIP_FORCE_RANGE since it's real
#: physics, not a training target.
_CRUSH_FORCE_RANGE = (0.1, 2000.0)

_MASS_CLASSES: tuple[MassClass, ...] = ("light", "medium", "heavy")


def _clamp(value: float, lo: float, hi: float, label: str) -> float:
    if value < lo or value > hi:
        clamped = min(max(value, lo), hi)
        LOGGER.warning("%s=%.6g out of range [%.6g, %.6g]; clamped to %.6g", label, value, lo, hi, clamped)
        return clamped
    return value


@dataclass
class ObjectParams:
    """Everything needed to build one robosuite object, plus grasp metadata."""

    name: str
    shape: Shape
    #: Half-extents (box: x,y,z), (cylinder: radius, half-height), or (ball: radius).
    size: tuple[float, ...] = (0.02, 0.02, 0.02)
    #: kg/m^3 -- robosuite's primitive objects take density directly, not mass.
    density: float = 1000.0
    #: (sliding, torsional, rolling) coefficients of friction.
    friction: tuple[float, float, float] = (0.5, 0.005, 0.0001)
    rgba: tuple[float, float, float, float] = field(default_factory=lambda: (0.5, 0.5, 0.5, 1.0))

    #: Coarse mass bucket. Descriptive only -- not used to derive `density`.
    mass_class: MassClass = "medium"
    #: Whether the object should be treated as breakable under excess grip force.
    fragile: bool = False
    #: Per-finger grip-force window a grasp should stay within: below
    #: `grip_force_min_N` the object would slip/drop, above `grip_force_max_N` it
    #: would be crushed. Metadata only for now -- see module docstring.
    grip_force_min_N: float = 1.0
    grip_force_max_N: float = 50.0

    #: Object stiffness k (N/m) for the spring-compression contact model
    #: `reaction_force_N` uses. See module docstring.
    spring_Npm: float = 1000.0
    #: Per-finger contact force (N) above which `rl/env.py`'s grip-force
    #: reward shaping treats the grasp as crushing. See module docstring.
    crush_force_N: float = 50.0

    def __post_init__(self) -> None:
        # JSON round-trips tuples back as lists; normalise so a config and its
        # reloaded snapshot compare equal (same reasoning as EnvConfig.obs_keys).
        self.size = tuple(float(s) for s in self.size)
        self.friction = tuple(float(f) for f in self.friction)
        self.rgba = tuple(float(c) for c in self.rgba)

        expected_dims = _SIZE_DIMS[self.shape]
        if len(self.size) != expected_dims:
            raise ValueError(f"shape={self.shape!r} needs {expected_dims} size value(s), got {self.size!r}")
        if len(self.friction) != 3:
            raise ValueError(f"friction needs 3 values (sliding, torsional, rolling), got {self.friction!r}")
        if self.mass_class not in _MASS_CLASSES:
            raise ValueError(f"mass_class must be one of {_MASS_CLASSES}, got {self.mass_class!r}")

        bounds = _SIZE_BOUNDS_M[self.shape]
        self.size = tuple(_clamp(s, *bounds[i], label=f"{self.name}.size[{i}]") for i, s in enumerate(self.size))
        self.density = _clamp(self.density, *_DENSITY_RANGE, label=f"{self.name}.density")
        self.friction = tuple(
            _clamp(f, *_FRICTION_BOUNDS[i], label=f"{self.name}.friction[{i}]") for i, f in enumerate(self.friction)
        )
        self.grip_force_min_N = _clamp(self.grip_force_min_N, *_GRIP_FORCE_RANGE, label=f"{self.name}.grip_force_min_N")
        self.grip_force_max_N = _clamp(self.grip_force_max_N, *_GRIP_FORCE_RANGE, label=f"{self.name}.grip_force_max_N")
        self.spring_Npm = _clamp(self.spring_Npm, *_SPRING_RANGE, label=f"{self.name}.spring_Npm")
        self.crush_force_N = _clamp(self.crush_force_N, *_CRUSH_FORCE_RANGE, label=f"{self.name}.crush_force_N")
        if self.grip_force_min_N >= self.grip_force_max_N:
            LOGGER.warning(
                "%s: grip_force_min_N (%.3g) >= grip_force_max_N (%.3g); widening max to min * 2",
                self.name,
                self.grip_force_min_N,
                self.grip_force_max_N,
            )
            self.grip_force_max_N = self.grip_force_min_N * 2

    @property
    def volume_m3(self) -> float:
        """Geometric volume implied by ``shape``/``size``."""
        if self.shape == "box":
            x, y, z = self.size
            return 8.0 * x * y * z  # size components are half-extents
        if self.shape == "cylinder":
            r, half_h = self.size
            return math.pi * r * r * (2.0 * half_h)
        if self.shape == "ball":
            (r,) = self.size
            return (4.0 / 3.0) * math.pi * r**3
        raise ValueError(f"Unknown shape: {self.shape!r}")  # unreachable given Shape/__post_init__

    @property
    def mass_kg(self) -> float:
        """Mass implied by ``density`` * ``volume_m3`` -- informational only."""
        return self.density * self.volume_m3

    @property
    def mass_g(self) -> float:
        """``mass_kg`` in grams -- used by `scripts/rollout_all_objects.py`'s summary table."""
        return self.mass_kg * 1000.0

    @property
    def rest_width_mm(self) -> float:
        """Natural (uncompressed) width across the narrowest graspable cross-section, in mm.

        For a box this is the shortest full-extent dimension (the axis a
        parallel-jaw gripper would actually close on); for a cylinder/ball it's
        the diameter. Used by `reaction_force_N`.
        """
        if self.shape == "box":
            width_m = min(2.0 * s for s in self.size)
        elif self.shape == "cylinder":
            radius, _ = self.size
            width_m = 2.0 * radius
        else:  # ball
            (radius,) = self.size
            width_m = 2.0 * radius
        return width_m * 1000.0

    def reaction_force_N(self, aperture_mm: float) -> float:
        """Spring-compression contact force (N) at a given gripper aperture.

        reaction = spring_Npm * max(0, rest_width_mm - aperture_mm) / 1000. Not
        currently read by `rl/env.py`, which uses a genuine MuJoCo fingertip force
        sensor instead (`objects/force_gripper.py`) -- see module docstring.
        """
        compression_mm = max(0.0, self.rest_width_mm - aperture_mm)
        return self.spring_Npm * compression_mm / 1000.0


#: Shape order used by `sample_object_params`/`object_params_to_z` -- fixed so a
#: one-hot position always means the same shape across every call/process.
_Z_SHAPES: tuple[Shape, ...] = tuple(_SIZE_DIMS)  # ("box", "cylinder", "ball")
_Z_MAX_SIZE_DIMS = max(_SIZE_DIMS.values())

#: one-hot shape (len(_Z_SHAPES)) + size zero-padded to _Z_MAX_SIZE_DIMS + density (1)
#: + friction (3). `mass_kg` is deliberately excluded: it's `density * volume_m3`, a
#: deterministic function of two dims already in z (density, size) -- feeding both
#: would just hand a FiLM/conditioning layer a redundant, perfectly-collinear input.
Z_DIM = len(_Z_SHAPES) + _Z_MAX_SIZE_DIMS + 1 + 3


def sample_object_params(shapes: Sequence[Shape] = _Z_SHAPES, name: str = "domain_random") -> ObjectParams:
    """Draw one :class:`ObjectParams` from a continuous, per-shape-bounded distribution.

    Domain-randomization sampler for the paradigm-switch baselines (pi_blind /
    pi_blind+hist / pi_param, see `objects/lift_object_task.ParamLift`): reuses the
    exact clamp ranges `__post_init__` already enforces (`_SIZE_BOUNDS_M`,
    `_DENSITY_RANGE`, `_FRICTION_BOUNDS`) as sampling ranges, so a sampled object is by
    construction never something `__post_init__` would have had to clamp.

    Uses the *global* `numpy.random` legacy API (`np.random.uniform`/`randint`), not a
    local `Generator` -- consistent with robosuite's own placement sampler, which also
    draws from the global RNG (see `rl/env.py`'s `reset()` docstring): this way
    `np.random.seed(seed)` there reproduces object sampling along with placement,
    instead of leaving a second, independently-seeded RNG stream.
    """
    shape = shapes[np.random.randint(len(shapes))]
    bounds = _SIZE_BOUNDS_M[shape]
    size = tuple(float(np.random.uniform(lo, hi)) for lo, hi in bounds)
    density = float(np.random.uniform(*_DENSITY_RANGE))
    friction = tuple(float(np.random.uniform(lo, hi)) for lo, hi in _FRICTION_BOUNDS)
    return ObjectParams(name=name, shape=shape, size=size, density=density, friction=friction)


def object_params_to_z(params: ObjectParams) -> np.ndarray:
    """Fixed-width, shape-agnostic physical-parameter vector (`Z_DIM`,), float32.

    Layout: one-hot shape | size (zero-padded to `_Z_MAX_SIZE_DIMS`) | density |
    friction (3). See `Z_DIM`'s docstring for why `mass_kg` is excluded. The one-hot
    shape block lets a FiLM/conditioning layer tell which padded size slots are
    meaningful for this object without a variable-width input.
    """
    shape_onehot = [1.0 if params.shape == s else 0.0 for s in _Z_SHAPES]
    size_padded = list(params.size) + [0.0] * (_Z_MAX_SIZE_DIMS - len(params.size))
    z = shape_onehot + size_padded + [params.density] + list(params.friction)
    return np.asarray(z, dtype=np.float32)


#: z-vector indices left exact by `object_params_to_noisy_z` -- the one-hot shape
#: block. Shape is assumed visually obvious (unlike density/friction, which have to be
#: inferred), so only indices from here on get perturbed.
_Z_NOISY_FROM = len(_Z_SHAPES)


def object_params_to_noisy_z(params: ObjectParams, rel_noise_std: float = 0.1) -> np.ndarray:
    """`object_params_to_z`, with multiplicative Gaussian noise on its continuous dims.

    Models a noisy sysID-style estimate of an object's physical parameters -- the
    input `pi_param` FiLM-conditions on, as opposed to `pi_blind`, which never sees z
    at all. `rel_noise_std=0.0` returns the exact (noiseless) z. Draws from the
    *global* `numpy.random` API, same reasoning as `sample_object_params`.
    """
    z = object_params_to_z(params)
    if rel_noise_std <= 0.0:
        return z
    noisy = z.copy()
    continuous = noisy[_Z_NOISY_FROM:]
    scale = 1.0 + np.random.normal(0.0, rel_noise_std, size=continuous.shape).astype(np.float32)
    noisy[_Z_NOISY_FROM:] = continuous * scale
    return noisy
