"""Generate a width-matched / mass-matched cylinder object set.

The 6 default objects (`configs/objects/prompts.json`) confound geometry and mass, so
this generates 5 cylinders that hold one axis fixed at a time:

    width40_mass050g   width=40mm, mass= 50g
    width40_mass200g   width=40mm, mass=200g   <- shared anchor point
    width40_mass500g   width=40mm, mass=500g
    width25_mass200g   width=25mm, mass=200g
    width55_mass200g   width=55mm, mass=200g

{width40_mass050g, width40_mass200g, width40_mass500g} isolates mass (width constant);
{width25_mass200g, width40_mass200g, width55_mass200g} isolates width (mass constant).
Cylinders, not balls, since a cylinder's half-height can vary independently to hit a
mass target without changing its radius (the gripper-relevant "width"). Every other
field is held constant across all 5. Deterministic and analytic (solved in closed
form from a target width/mass), not LLM-extracted.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/generate_width_mass_objects.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path

from objects.object_params import ObjectParams
from common.utils import setup_logging

LOGGER = logging.getLogger(__name__)

#: Anchor point shared by both groups: width40_mass200g. Half-height chosen so every
#: object in both groups stays within ObjectParams' clamp ranges.
_ANCHOR_WIDTH_MM = 40.0
_ANCHOR_MASS_G = 200.0
_ANCHOR_HALF_HEIGHT_M = 0.045

#: Width-matched group: radius (and thus density, solved at the anchor) held fixed;
#: half-height solved per target mass.
_WIDTH_MATCHED_MASSES_G: tuple[float, ...] = (50.0, 200.0, 500.0)
#: Mass-matched group: half-height (and mass) held fixed; density solved per target width.
_MASS_MATCHED_WIDTHS_MM: tuple[float, ...] = (25.0, 40.0, 55.0)

#: Held constant across all 5 objects -- only width/mass vary between them.
_SHARED_FIELDS: dict[str, object] = {
    "friction": (0.5, 0.005, 0.0001),
    "rgba": (0.6, 0.75, 0.85, 1.0),
    "fragile": True,
    "grip_force_min_N": 2.0,
    "grip_force_max_N": 10.0,
    "spring_Npm": 3000.0,
    "crush_force_N": 10.0,
}


def _mass_class_for(mass_g: float) -> str:
    """Descriptive-only bucket (see ObjectParams.mass_class) matching the rough
    light/medium/heavy scale the existing 6 narrative objects use."""
    if mass_g < 100.0:
        return "light"
    if mass_g < 300.0:
        return "medium"
    return "heavy"


def _cylinder_density(radius_m: float, half_height_m: float, mass_kg: float) -> float:
    """Solve density (kg/m^3) for a cylinder of given radius/half-height/mass."""
    volume_m3 = math.pi * radius_m**2 * (2.0 * half_height_m)
    return mass_kg / volume_m3


def _cylinder_half_height(radius_m: float, density: float, mass_kg: float) -> float:
    """Solve half-height (m) for a cylinder of given radius/density/mass."""
    return mass_kg / (density * math.pi * radius_m**2 * 2.0)


def _make_object(name: str, width_mm: float, mass_g: float, *, radius_m: float, half_height_m: float, density: float) -> ObjectParams:
    params = ObjectParams(
        name=name,
        shape="cylinder",
        size=(radius_m, half_height_m),
        density=density,
        mass_class=_mass_class_for(mass_g),
        **_SHARED_FIELDS,
    )
    # Sanity check: the closed-form solve should reproduce the requested width/mass
    # to within float error, assuming no clamp fired (ObjectParams warns on its own
    # if one did -- see object_params.py's _clamp).
    got_width_mm = params.rest_width_mm
    got_mass_g = params.mass_g
    if abs(got_width_mm - width_mm) > 0.05 or abs(got_mass_g - mass_g) > 0.5:
        raise AssertionError(
            f"{name}: solved size/density didn't reproduce the target -- wanted "
            f"width={width_mm}mm/mass={mass_g}g, got width={got_width_mm:.3f}mm/"
            f"mass={got_mass_g:.3f}g. A clamp likely fired; see the warning above."
        )
    return params


def generate_objects() -> list[ObjectParams]:
    anchor_radius_m = _ANCHOR_WIDTH_MM / 2.0 / 1000.0
    anchor_density = _cylinder_density(anchor_radius_m, _ANCHOR_HALF_HEIGHT_M, _ANCHOR_MASS_G / 1000.0)

    objects: list[ObjectParams] = []

    for mass_g in _WIDTH_MATCHED_MASSES_G:
        half_height_m = _cylinder_half_height(anchor_radius_m, anchor_density, mass_g / 1000.0)
        name = f"width{_ANCHOR_WIDTH_MM:.0f}_mass{mass_g:03.0f}g"
        objects.append(
            _make_object(
                name,
                _ANCHOR_WIDTH_MM,
                mass_g,
                radius_m=anchor_radius_m,
                half_height_m=half_height_m,
                density=anchor_density,
            )
        )

    for width_mm in _MASS_MATCHED_WIDTHS_MM:
        if width_mm == _ANCHOR_WIDTH_MM:
            continue  # width40_mass200g already generated above -- it's the shared anchor
        radius_m = width_mm / 2.0 / 1000.0
        density = _cylinder_density(radius_m, _ANCHOR_HALF_HEIGHT_M, _ANCHOR_MASS_G / 1000.0)
        name = f"width{width_mm:.0f}_mass{_ANCHOR_MASS_G:03.0f}g"
        objects.append(
            _make_object(
                name,
                width_mm,
                _ANCHOR_MASS_G,
                radius_m=radius_m,
                half_height_m=_ANCHOR_HALF_HEIGHT_M,
                density=density,
            )
        )

    return objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("src/configs/objects/width_mass_set")
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for params in generate_objects():
        out_path = args.out_dir / f"{params.name}.json"
        out_path.write_text(json.dumps(dataclasses.asdict(params), indent=2))
        LOGGER.info(
            "%-20s -> %s | width=%.1fmm mass=%.1fg density=%.1f half_height=%.4fm",
            params.name,
            out_path,
            params.rest_width_mm,
            params.mass_g,
            params.density,
            params.size[1],
        )

    LOGGER.info("Wrote 5 object snapshot(s) to %s", args.out_dir)


if __name__ == "__main__":
    main()
