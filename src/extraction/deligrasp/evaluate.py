"""
Ground-truth evaluation of a grasp outcome.

Given the *true* object physics and the force actually applied by the gripper,
decide whether the grasp:
    - crushed  : peak contact force exceeded the object's crush threshold
    - dropped  : final contact force was below what is needed to hold it
    - held     : enough force to hold, without crushing  (SUCCESS)
"""

from dataclasses import dataclass


@dataclass
class Outcome:
    object_name: str
    method: str
    outcome: str          # "held" | "dropped" | "crushed"
    success: bool
    final_force_N: float
    peak_force_N: float
    required_force_N: float
    crush_force_N: float

    def row(self):
        return (f"{self.object_name:<14} {self.method:<12} {self.outcome:<9} "
                f"req={self.required_force_N:5.2f}N  "
                f"applied={self.final_force_N:5.2f}N  "
                f"peak={self.peak_force_N:5.2f}N  "
                f"crush={self.crush_force_N:6.1f}N  "
                f"{'OK' if self.success else 'X'}")


def _true_contact_force(obj, aperture_mm, force_limit_N):
    """The force a real load cell would read for this object at this aperture."""
    compression_m = max(0.0, (obj.rest_width_mm - aperture_mm)) / 1000.0
    reaction = obj.spring_Npm * compression_m
    return min(reaction, force_limit_N)


def evaluate(obj, method, final_aperture_mm, applied_force_N, peak_force_N):
    final = _true_contact_force(obj, final_aperture_mm, applied_force_N)
    required = obj.required_force_N
    crush = obj.crush_force_N

    eps = 1e-6
    if peak_force_N > crush + eps:
        outcome, success = "crushed", False
    elif final + eps < required:
        outcome, success = "dropped", False
    else:
        outcome, success = "held", True

    return Outcome(obj.name, method, outcome, success,
                   round(final, 3), round(peak_force_N, 3),
                   round(required, 3), round(crush, 3))
