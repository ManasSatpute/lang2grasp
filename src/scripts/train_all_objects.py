"""Local dev driver: train a SAC policy for every object snapshot, one
after another, in the same process. For real cluster runs use a SLURM job
array instead -- this script is for quick iteration on a single machine.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/train_all_objects.py --base-config src/configs/policy/sac.json
    PYTHONPATH=src python src/scripts/train_all_objects.py --total-timesteps 2000  # smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rl.config import load_config
from objects.object_params import ObjectParams
from rl.train import train
from common.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path, default=Path("src/configs/objects"))
    parser.add_argument("--base-config", type=Path, default=Path("src/configs/policy/sac.json"))
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    object_files = sorted(p for p in args.objects_dir.glob("*.json") if p.name != "prompts.json")
    if not object_files:
        raise SystemExit(
            f"No object snapshots found in {args.objects_dir}. "
            "Run scripts/extract_object_params.py first."
        )

    results: dict[str, Path] = {}
    for path in object_files:
        params = ObjectParams(**json.loads(path.read_text()))
        overrides = {
            "total_timesteps": args.total_timesteps,
            "log_dir": args.log_dir,
            "seed": args.seed,
            "run_name": f"lift_{params.name}",
        }
        cfg = load_config(args.base_config, overrides=overrides)
        cfg.env.object = params

        LOGGER.info("=== Training %s ===", params.name)
        run_dir, _ = train(cfg)
        results[params.name] = run_dir

    LOGGER.info("All objects trained:")
    for name, run_dir in results.items():
        LOGGER.info("  %-16s -> %s", name, run_dir)


if __name__ == "__main__":
    main()
