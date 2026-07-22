"""Stage 3: roll every trained per-object policy out against the Panda arm in
robosuite/MuJoCo, and print a per-object success-rate table.

For a single run, `python -m rl.rollout --run-dir ...` still works
unchanged -- this script just loops that same `rollout()` over every trained
object's run directory.

Each run's `config.json` snapshot already carries the resolved `env.object`
(None for the stock-cube baseline, an `ObjectParams` dict for every
LLM-described object) -- read straight from that instead of re-reading
`src/configs/objects/*.json`, so the table reflects what a run actually
trained against even if the source snapshot has since changed. `mass_g` /
`rest_width_mm` are pulled in to make the mass/size-vs-success relationship
the array run is meant to surface directly readable, without a separate
analysis pass.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.rollout import rollout
from objects.object_params import ObjectParams
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


def _load_object_params(run_dir: Path) -> ObjectParams | None:
    """None for the stock-cube baseline (`env.object` unset at train time)."""
    config = json.loads((run_dir / CONFIG_SNAPSHOT).read_text())
    object_payload = config.get("env", {}).get("object")
    return ObjectParams(**object_payload) if object_payload else None


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
        object_params = _load_object_params(run_dir)
        metrics = rollout(run_dir, episodes=args.episodes, seed=args.seed, device=args.device)
        rows.append(
            (
                object_name,
                metrics["success_rate"],
                metrics["mean_return"],
                metrics["n_episodes"],
                object_params.mass_g if object_params else None,
                object_params.rest_width_mm if object_params else None,
            )
        )

    # Baseline (mass_g=None, stock cube) sorts last; everything else ascending
    # by mass so the mass/size-vs-success trend the array run is testing for
    # reads straight off the table.
    rows.sort(key=lambda r: (r[4] is None, r[4]))

    header = (
        f"{'object':<24} {'success_rate':>12} {'mean_return':>12} {'episodes':>9} "
        f"{'mass_g':>10} {'width_mm':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, success_rate, mean_return, n, mass_g, width_mm in rows:
        mass_str = f"{mass_g:.1f}" if mass_g is not None else "-"
        width_str = f"{width_mm:.1f}" if width_mm is not None else "-"
        print(
            f"{name:<24} {success_rate:>12.2f} {mean_return:>12.2f} {n:>9d} "
            f"{mass_str:>10} {width_str:>10}"
        )


if __name__ == "__main__":
    main()
