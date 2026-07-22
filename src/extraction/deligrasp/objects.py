"""
Ground-truth physical objects for the DeliGrasp simulation.

These are the same 6 objects, described by the same :class:`ObjectParams`
(`objects/object_params.py`), as the robosuite/SAC training pipeline uses --
`extraction/param_prompts.py`'s ``PRIORS`` (and its on-disk snapshot,
`configs/objects/*.json`) is the single ground-truth source both benchmarks
draw from. Two fields exist only for this benchmark's physics -- see
``ObjectParams``'s module docstring:

    spring_Npm      object stiffness k (N/m); soft ~20, very stiff ~10000
    crush_force_N   per-finger normal contact force at which the object is damaged

The values used to decide whether a grasp holds, drops, or crushes the object
come straight from that shared ``ObjectParams`` (``mass_g``, ``friction``,
``rest_width_mm``, ``required_force_N`` -- all derived from
shape/size/density/friction). The LLM never sees any of this -- it must infer
comparable quantities from the object's name/description alone
(`prompts.LLM_PRIORS`, kept deliberately separate). That gap is the whole
point of the experiment.
"""

from extraction.param_prompts import PRIORS
from objects.object_params import ObjectParams

# The 6 objects from `extraction/param_prompts.py` / `configs/objects/prompts.json`,
# so this benchmark and the object-parameter extraction pipeline describe the same
# objects with the same physics.
#   - fragile & rigid   (glass_bottle, ceramic_mug, raw_egg): crack/shatter, need a
#     narrow, low grip-force window
#   - rugged & tough    (steel_bolt, rice_bag, brick): high grip-force tolerance
BENCHMARK: dict[str, ObjectParams] = {name: ObjectParams(name=name, **fields) for name, fields in PRIORS.items()}


def get(name: str) -> ObjectParams:
    return BENCHMARK[name]
