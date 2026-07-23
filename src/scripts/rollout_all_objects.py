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

Besides the console table, this always writes a CSV of the full metric set
(`--results-dir`, default `src/results/`). `--plot` additionally saves a
success-rate/return chart, and `--video` renders a short rollout video per
object (its own short episode count, `--video-episodes`, independent of
`--episodes` used for the stats, so a handful of objects doesn't turn into
many minutes of footage).

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/rollout_all_objects.py --runs-dir runs --episodes 20
    PYTHONPATH=src python src/scripts/rollout_all_objects.py --plot --video
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rl.rollout import rollout
from objects.object_params import ObjectParams
from common.utils import CONFIG_SNAPSHOT, setup_logging

#: Full per-object metric set written to the CSV -- a superset of what the
#: console table prints (see main()).
_CSV_FIELDS = (
    "object",
    "success_rate",
    "mean_return",
    "std_return",
    "mean_length",
    "success_ever",
    "success_stderr",
    "n_episodes",
    "mass_g",
    "rest_width_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Directory containing lift_<object> run dirs."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("src/results"),
        help="Where to write the results CSV, plot, and (with --video) rollout videos.",
    )
    parser.add_argument("--plot", action="store_true", help="Save a success-rate/return plot.")
    parser.add_argument("--video", action="store_true", help="Save a short rollout video per object.")
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=3,
        help="Episodes to render per object when --video is set. Kept small: offscreen "
        "rendering is much slower than physics-only stepping.",
    )
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

    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in run_dirs:
        object_name = run_dir.name.removeprefix("lift_")
        object_params = _load_object_params(run_dir)
        metrics = rollout(run_dir, episodes=args.episodes, seed=args.seed, device=args.device)
        rows.append(
            {
                "object": object_name,
                **metrics,
                "mass_g": object_params.mass_g if object_params else None,
                "rest_width_mm": object_params.rest_width_mm if object_params else None,
            }
        )

        if args.video:
            video_dir = args.results_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            rollout(
                run_dir,
                episodes=args.video_episodes,
                seed=args.seed,
                device=args.device,
                video_path=video_dir / f"{object_name}.mp4",
            )

    # Baseline (mass_g=None, stock cube) sorts last; everything else ascending
    # by mass so the mass/size-vs-success trend the array run is testing for
    # reads straight off the table.
    rows.sort(key=lambda r: (r["mass_g"] is None, r["mass_g"]))

    header = (
        f"{'object':<24} {'success_rate':>12} {'mean_return':>12} {'episodes':>9} "
        f"{'mass_g':>10} {'width_mm':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        mass_str = f"{r['mass_g']:.1f}" if r["mass_g"] is not None else "-"
        width_str = f"{r['rest_width_mm']:.1f}" if r["rest_width_mm"] is not None else "-"
        print(
            f"{r['object']:<24} {r['success_rate']:>12.2f} {r['mean_return']:>12.2f} "
            f"{r['n_episodes']:>9d} {mass_str:>10} {width_str:>10}"
        )

    csv_path = args.results_dir / "rollout_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in _CSV_FIELDS})
    print(f"\nWrote {csv_path}")

    if args.plot:
        from plot_rollout_results import plot_results

        plot_results(rows, path=args.results_dir / "rollout_success_rate.png")


if __name__ == "__main__":
    main()
