"""Stage 3: roll every trained per-object policy out against the Panda arm in
robosuite/MuJoCo, and print a per-object success-rate table.

For a single run, `python -m rl.rollout --run-dir ...` still works
unchanged -- this script just loops that same `rollout()` over every trained
object's run directory.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.rollout import rollout
from common.utils import CONFIG_SNAPSHOT, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Directory containing lift_<object> run dirs."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    run_dirs = sorted(
        d for d in args.runs_dir.glob("lift_*") if d.is_dir() and (d / CONFIG_SNAPSHOT).exists()
    )
    if not run_dirs:
        raise SystemExit(f"No trained runs found under {args.runs_dir}")

    rows = []
    for run_dir in run_dirs:
        object_name = run_dir.name.removeprefix("lift_")
        metrics = rollout(run_dir, episodes=args.episodes, seed=args.seed, device=args.device)
        rows.append((object_name, metrics["success_rate"], metrics["mean_return"], metrics["n_episodes"]))

    header = f"{'object':<24} {'success_rate':>12} {'mean_return':>12} {'episodes':>9}"
    print(header)
    print("-" * len(header))
    for name, success_rate, mean_return, n in rows:
        print(f"{name:<24} {success_rate:>12.2f} {mean_return:>12.2f} {n:>9d}")


if __name__ == "__main__":
    main()
