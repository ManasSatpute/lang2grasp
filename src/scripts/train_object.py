"""Stage 2: train one SAC policy for a single LLM-parameterised object.

A thin wrapper around `rl.train.train` -- loads an `ObjectParams` snapshot (the
*extracted* object, from `scripts/extract_object_params.py`), and, when a golden
(ground-truth) entry exists for that name (`extraction.param_prompts.
golden_object_params`), splits it in two: the golden object is what's physically built
into the scene (`EnvConfig.object`), while the extracted object drives crush
behaviour, the grip-force bonus, and the z-vector (`EnvConfig.extracted_object`). Falls
back to the old coupled behaviour for names with no golden entry (e.g. `width_mass_set`
objects). See `rl/env.py`'s module docstring, "Golden vs. extracted object".

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/train_object.py \\
        --object src/configs/objects/raw_egg.json --base-config src/configs/policy/sac.json
    PYTHONPATH=src python src/scripts/train_object.py \\
        --object src/configs/objects/raw_egg.json --base-config src/configs/policy/sac.json \\
        --total-timesteps 2000   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import logging

from rl.config import TRAIN_OVERRIDE_FIELDS, add_override_args, load_config
from extraction.param_prompts import golden_object_params
from objects.object_params import ObjectParams
from rl.train import train
from common.utils import EXIT_REQUEUE, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", type=Path, required=True, help="Path to a configs/objects/<name>.json snapshot."
    )
    parser.add_argument("--base-config", type=Path, default=Path("src/configs/policy/sac.json"))
    # Mirrors rl.train's CLI overrides, so a per-object run tunes like any other.
    add_override_args(parser)
    parser.add_argument(
        "--grip-force-shaping",
        action="store_true",
        help="Enable EnvConfig.grip_force_shaping (off by default): a reward bonus "
        "for holding within grip_force_min_N/max_N. Does not affect crush behaviour, "
        "which is unconditional whenever --object is set.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    extracted_params = ObjectParams(**json.loads(args.object.read_text()))
    golden_params = golden_object_params(extracted_params.name)
    if golden_params is None:
        LOGGER.info(
            "No golden entry for %r; physics and perception both use the extracted "
            "params (old, coupled behaviour).",
            extracted_params.name,
        )
    overrides = {key: getattr(args, key) for key in TRAIN_OVERRIDE_FIELDS}

    cfg = load_config(args.base_config, overrides=overrides)
    cfg.env.object = golden_params or extracted_params
    cfg.env.extracted_object = extracted_params if golden_params is not None else None
    cfg.env.grip_force_shaping = args.grip_force_shaping
    if cfg.run_name is None:
        cfg.run_name = f"lift_{extracted_params.name}"

    _, needs_requeue = train(cfg)
    # Same exit-code contract as rl.train.main(): a SLURM job-array driver
    # reads this to decide whether to requeue.
    sys.exit(EXIT_REQUEUE if needs_requeue else 0)


if __name__ == "__main__":
    main()
