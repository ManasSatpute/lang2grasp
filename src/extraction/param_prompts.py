"""Prompt template, JSON schema, and offline priors for object-parameter extraction.

The 6 default objects (also in ``configs/objects/prompts.json``) span the axes
that matter for grasping -- fragile vs. rugged, light vs. heavy, slick vs.
grippy -- so the SAC policy sees meaningfully different object dynamics rather
than six near-identical boxes:

    glass_bottle  fragile, light,  low safe grip force
    steel_bolt    rugged,  light,  high grip force tolerated
    ceramic_mug   fragile, medium
    rice_bag      rugged,  medium, doesn't care about grip force
    raw_egg       extremely fragile, very light, narrow safe force window
    brick         rugged,  heavy
"""

from __future__ import annotations

import re

#: robosuite/MuJoCo units throughout: metres, kg/m^3 density, Newtons.
SYSTEM_PROMPT = """You estimate physical simulation and grasp parameters for an \
everyday object, for a robot-arm grasping simulator (MuJoCo/robosuite).

Given a short description of an object, output:
- shape: the closest primitive -- "box", "cylinder", or "ball"
- size: half-extents in metres. box needs 3 values [x, y, z]; cylinder needs 2 \
[radius, half_height]; ball needs 1 [radius].
- density: kg/m^3. Solid steel is ~7850; wood ~700; a thin-walled/hollow object \
(glass bottle, ceramic mug) is much lower (~300-900) than its bulk material \
density because most of its volume is empty space.
- friction: [sliding, torsional, rolling] coefficients. Sliding is the main one, \
roughly 0.05 (slick/wet) to 1.5 (rubbery/grippy); torsional and rolling are much \
smaller (roughly 0.001-0.01).
- mass_class: "light", "medium", or "heavy".
- fragile: true if the object would crack, shatter, or crush under excess grip \
force; false for rugged/tough objects.
- grip_force_min_N: minimum per-finger force (Newtons) needed to hold the object \
without it slipping.
- grip_force_max_N: maximum per-finger force (Newtons) the object can take before \
it is damaged. For fragile objects this should be close to grip_force_min_N \
(a narrow safe window); for rugged objects it can be much larger.
- rgba: approximate colour as [r, g, b, a], each 0-1. Use [0.5, 0.5, 0.5, 1.0] if \
the object's colour is not implied by its description.
- spring_Npm: object stiffness (Newtons per metre), for a spring-compression grasp \
model. Roughly 20 N/m for a very soft/compliant object up to 10000 N/m for a rigid \
one (steel, ceramic, brick).
- crush_force_N: per-finger contact force (Newtons) that actually breaks the \
object -- the real physical damage threshold, distinct from grip_force_max_N \
(a friendlier training-time target). Can be much larger than grip_force_max_N.

Respond with only the JSON object, no other text."""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "shape": {"type": "string", "enum": ["box", "cylinder", "ball"]},
        "size": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Half-extents in metres: 3 for box, 2 for cylinder, 1 for ball.",
        },
        "density": {"type": "number", "description": "kg/m^3."},
        "friction": {
            "type": "array",
            "items": {"type": "number"},
            "description": "[sliding, torsional, rolling] coefficients of friction.",
        },
        "mass_class": {"type": "string", "enum": ["light", "medium", "heavy"]},
        "fragile": {"type": "boolean"},
        "grip_force_min_N": {"type": "number"},
        "grip_force_max_N": {"type": "number"},
        "rgba": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Approximate [r, g, b, a] color, each in 0-1.",
        },
        "spring_Npm": {"type": "number", "description": "Stiffness in Newtons per metre."},
        "crush_force_N": {"type": "number", "description": "Per-finger force (N) that breaks the object."},
    },
    "required": [
        "shape",
        "size",
        "density",
        "friction",
        "mass_class",
        "fragile",
        "grip_force_min_N",
        "grip_force_max_N",
        "rgba",
        "spring_Npm",
        "crush_force_N",
    ],
    "additionalProperties": False,
}

#: name -> raw fields dict, matching JSON_SCHEMA. Ground-truth-ish priors for
#: MockBackend, the offline/deterministic default extraction backend.
PRIORS: dict[str, dict] = {
    "glass_bottle": {
        "shape": "cylinder",
        "size": [0.025, 0.09],
        "density": 300.0,  # thin-walled/hollow -> light despite being glass
        "friction": [0.25, 0.005, 0.0001],  # slippery glass surface
        "mass_class": "light",
        "fragile": True,
        "grip_force_min_N": 2.0,
        "grip_force_max_N": 8.0,
        "rgba": [0.7, 0.85, 0.8, 0.5],
        "spring_Npm": 4000.0,
        "crush_force_N": 8.0,
    },
    "steel_bolt": {
        "shape": "cylinder",
        "size": [0.008, 0.03],
        "density": 7850.0,  # real steel density
        "friction": [0.9, 0.01, 0.001],  # rough, grips well
        "mass_class": "light",  # small object, low absolute mass despite density
        "fragile": False,
        "grip_force_min_N": 10.0,
        "grip_force_max_N": 60.0,
        "rgba": [0.6, 0.6, 0.65, 1.0],
        "spring_Npm": 6000.0,
        "crush_force_N": 150.0,
    },
    "ceramic_mug": {
        "shape": "cylinder",
        "size": [0.04, 0.05],
        "density": 900.0,
        "friction": [0.45, 0.005, 0.0001],
        "mass_class": "medium",
        "fragile": True,
        "grip_force_min_N": 3.0,
        "grip_force_max_N": 12.0,
        "rgba": [0.95, 0.93, 0.85, 1.0],
        "spring_Npm": 3500.0,
        "crush_force_N": 12.0,
    },
    "rice_bag": {
        "shape": "box",
        "size": [0.06, 0.04, 0.02],
        "density": 750.0,
        "friction": [0.8, 0.01, 0.001],  # cloth/plastic bag, grips easily
        "mass_class": "medium",
        "fragile": False,
        "grip_force_min_N": 8.0,
        "grip_force_max_N": 50.0,
        "rgba": [0.85, 0.8, 0.65, 1.0],
        "spring_Npm": 300.0,
        "crush_force_N": 500.0,
    },
    "raw_egg": {
        "shape": "ball",
        "size": [0.02],
        "density": 550.0,
        "friction": [0.2, 0.002, 0.0001],  # smooth shell, low grip margin
        "mass_class": "light",
        "fragile": True,
        "grip_force_min_N": 1.0,
        "grip_force_max_N": 4.0,  # very narrow safe range
        "rgba": [0.95, 0.93, 0.85, 1.0],
        "spring_Npm": 3000.0,
        "crush_force_N": 5.0,
    },
    "brick": {
        "shape": "box",
        "size": [0.05, 0.03, 0.02],
        "density": 1900.0,
        "friction": [0.7, 0.01, 0.001],
        "mass_class": "heavy",
        "fragile": False,
        "grip_force_min_N": 15.0,
        "grip_force_max_N": 80.0,
        "rgba": [0.6, 0.3, 0.25, 1.0],
        "spring_Npm": 8000.0,
        "crush_force_N": 1000.0,
    },
}

#: shape hint keywords for the generic fallback (unrecognised object names).
_SHAPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ball": ("ball", "sphere", "orb", "marble", "egg"),
    "cylinder": ("can", "bottle", "cylinder", "tube", "cup", "mug", "jar", "roll", "bolt"),
}


def prior_for(prompt: str) -> tuple[str, dict]:
    """Match ``prompt`` against known objects, or fall back to a shape-keyword guess.

    Returns ``(matched_key, fields)`` where ``fields`` is a raw dict matching
    :data:`JSON_SCHEMA` -- the same shape :class:`ParamBackend.extract` returns.
    """
    text = prompt.lower()
    for key, fields in PRIORS.items():
        # Match on constituent words rather than the exact phrase, so "a small
        # glass bottle of wine" still matches the "glass_bottle" prior.
        words = key.split("_")
        if all(re.search(rf"\b{re.escape(w)}\b", text) for w in words):
            return key, dict(fields)

    shape = "box"
    for candidate_shape, keywords in _SHAPE_KEYWORDS.items():
        if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
            shape = candidate_shape
            break

    size = {"box": (0.03, 0.03, 0.03), "cylinder": (0.03, 0.05), "ball": (0.03,)}[shape]
    return "generic", {
        "shape": shape,
        "size": list(size),
        "density": 1000.0,
        "friction": [0.5, 0.005, 0.0001],
        "mass_class": "medium",
        "fragile": False,
        "grip_force_min_N": 5.0,
        "grip_force_max_N": 30.0,
        "rgba": [0.5, 0.5, 0.5, 1.0],
        "spring_Npm": 1000.0,
        "crush_force_N": 50.0,
    }
