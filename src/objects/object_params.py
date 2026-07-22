"""Physical + grasp-relevant parameters for an LLM-described object.

Two groups of fields:

- **Simulation fields** (``shape``, ``size``, ``density``, ``friction``) map directly
  onto robosuite's primitive objects (``BoxObject``/``CylinderObject``/``BallObject``,
  see ``lift_object_task.build_mujoco_object``), which take exactly this shape/size/
  density/friction quartet. ``friction`` is robosuite's own 3-tuple: (sliding,
  torsional, rolling).
- **Grasp-descriptive fields** (``mass_class``, ``fragile``, ``grip_force_min_N``,
  ``grip_force_max_N``) are *metadata* carried through the pipeline (extraction ->
  training config -> rollout output) for now -- they do not currently change the SAC
  reward or the physics. They're the natural hook for a future force-adaptive grasp
  reward (holding a fragile object too hard = crushed, too soft = dropped), the same
  idea `extraction/deligrasp` explores standalone, but wiring that into training is a
  separate step from defining the schema.
- **DeliGrasp fields** (``spring_Npm``, ``crush_force_N``) exist purely for
  `extraction/deligrasp`'s spring-compression grasp simulator (see
  `extraction/deligrasp/gripper.py` / `evaluate.py`). They're not used by the
  robosuite/SAC path at all. Unlike ``grip_force_max_N`` (a friendly training-time
  target range), ``crush_force_N`` is the literal contact force at which
  `evaluate.py` marks a grasp "crushed" -- the two can and do differ in scale for
  the same object.

Values are clamped to ranges that stay graspable by a Panda parallel-jaw gripper and
numerically stable in MuJoCo. An LLM extrapolating from a text prompt occasionally
returns something wild (a "boulder" at 50 kg, a friction of 9) -- clamping here, once,
means every caller (env, tests, rollout) sees only safe values.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

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
#: N/m, DeliGrasp-only. 20 ~ a very soft/compliant object, 10000 ~ a rigid one
#: (brick-like); see `extraction/deligrasp/prompts.py`'s PROMPT_THINKER, which
#: quotes the same 20-2000 range to the LLM as a rule of thumb.
_SPRING_RANGE = (20.0, 10000.0)
#: Newtons, per finger, DeliGrasp-only -- the literal contact force that breaks
#: the object in the spring-compression simulator. Deliberately a much wider
#: range than _GRIP_FORCE_RANGE since it's real physics, not a training target.
_CRUSH_FORCE_RANGE = (0.1, 2000.0)

_MASS_CLASSES: tuple[MassClass, ...] = ("light", "medium", "heavy")
#: m/s^2, for `required_force_N`.
_G = 9.81


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

    #: Object stiffness k (N/m) for `extraction/deligrasp`'s spring-compression
    #: contact model. DeliGrasp-only, see module docstring.
    spring_Npm: float = 1000.0
    #: Per-finger contact force (N) above which `extraction/deligrasp/evaluate.py`
    #: scores the grasp "crushed". DeliGrasp-only, see module docstring.
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
        """``mass_kg`` in grams -- the unit `extraction/deligrasp` works in."""
        return self.mass_kg * 1000.0

    @property
    def rest_width_mm(self) -> float:
        """Natural (uncompressed) width across the narrowest graspable cross-section, in mm.

        For a box this is the shortest full-extent dimension (the axis a
        parallel-jaw gripper would actually close on); for a cylinder/ball it's
        the diameter. DeliGrasp-only, see module docstring.
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

    @property
    def required_force_N(self) -> float:
        """Minimum per-finger normal force needed to hold the object statically.

        Static equilibrium for a symmetric two-finger grasp:
            2 * mu * N  >=  m * g     ->     N >= m*g / (2*mu)
        Uses the sliding-friction component of ``friction``. DeliGrasp-only.
        """
        return (self.mass_g / 1000.0) * _G / (2.0 * self.friction[0])
