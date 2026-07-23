"""Stage 2: train one SAC policy for a single LLM-parameterised object.

A thin wrapper around `rl.train.train` -- loads an `ObjectParams`
snapshot (from `scripts/extract_object_params.py`), sets it on the env config,
and delegates to the exact same checkpoint/resume/requeue machinery
`rl.train` already provides for the un-parameterised baseline.

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

from rl.config import TRAIN_OVERRIDE_FIELDS, add_override_args, load_config
from objects.object_params import ObjectParams
from rl.train import train
from common.utils import EXIT_REQUEUE, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object", type=Path, required=True, help="Path to a configs/objects/<name>.json snapshot."
    )
    parser.add_argument("--base-config", type=Path, default=Path("src/configs/policy/sac.json"))
    # Mirrors rl.train's CLI overrides, so a per-object run tunes like any other.
    add_override_args(parser)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    object_params = ObjectParams(**json.loads(args.object.read_text()))
    overrides = {key: getattr(args, key) for key in TRAIN_OVERRIDE_FIELDS}

    cfg = load_config(args.base_config, overrides=overrides)
    cfg.env.object = object_params
    if cfg.run_name is None:
        cfg.run_name = f"lift_{object_params.name}"

    _, needs_requeue = train(cfg)
    # Same exit-code contract as rl.train.main(): a SLURM job-array driver
    # reads this to decide whether to requeue.
    sys.exit(EXIT_REQUEUE if needs_requeue else 0)


if __name__ == "__main__":
    main()
