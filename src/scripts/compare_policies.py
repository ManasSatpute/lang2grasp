"""Compare a generic (no-extracted-params) policy against the per-object pipeline.

The "generic" arm is a single SAC policy trained once with `EnvConfig.object` unset
(the existing baseline path -- no `--object` flag, e.g.
`python -m rl.train --config src/configs/policy/sac.json --run-name lift_baseline`).
It never sees any object's extracted shape/size/density/friction, and its reward
never uses grip-force parameters either (`grip_force_shaping` requires `object` to be
set). This script rolls that one policy out against each LLM-described object's
*actual* physics (`rl.rollout.rollout`'s `object_override`) and compares it, object by
object, against that object's own dedicated policy from `scripts/train_object.py` /
`scripts/train_all_objects.py`.

Usage (from the repo root):
    PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json --run-name lift_baseline
    PYTHONPATH=src python src/scripts/train_all_objects.py --base-config src/configs/policy/sac.json
    PYTHONPATH=src python src/scripts/compare_policies.py --baseline-run-dir runs/lift_baseline --plot
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from rl.rollout import load_policy, load_run_object_params, rollout
from common.utils import CONFIG_SNAPSHOT, FINAL_MODEL_NAME, setup_logging

LOGGER = logging.getLogger(__name__)

_CSV_FIELDS = (
    "object",
    "generic_success_rate",
    "generic_mean_return",
    "generic_std_return",
    "generic_n_episodes",
    "object_aware_success_rate",
    "object_aware_mean_return",
    "object_aware_std_return",
    "object_aware_n_episodes",
    "mass_g",
    "rest_width_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        required=True,
        help="Run dir of the generically-trained policy (env.object unset at train time).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing lift_<object> run dirs (the per-object policies).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("src/results"))
    parser.add_argument("--plot", action="store_true", help="Save a grouped comparison plot.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    run_dirs = sorted(
        d for d in args.runs_dir.glob("lift_*") if d.is_dir() and (d / CONFIG_SNAPSHOT).exists()
    )
    if not run_dirs:
        raise SystemExit(f"No per-object runs found under {args.runs_dir}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, baseline_cfg = load_policy(args.baseline_run_dir, FINAL_MODEL_NAME, args.device)
    if baseline_cfg.env.object is not None:
        LOGGER.warning(
            "Baseline run %s was trained with env.object=%s set -- it isn't actually "
            "generic. Pass a run trained without --object.",
            args.baseline_run_dir,
            baseline_cfg.env.object.name,
        )

    rows = []
    for run_dir in run_dirs:
        object_params = load_run_object_params(run_dir)
        if object_params is None:
            # The stock-cube baseline itself, or another run with no env.object --
            # there's no per-object physics here to compare the generic policy against.
            LOGGER.info("Skipping %s: no env.object in its config snapshot.", run_dir)
            continue
        object_name = run_dir.name.removeprefix("lift_")

        aware_metrics = rollout(run_dir, episodes=args.episodes, seed=args.seed, device=args.device)
        generic_metrics = rollout(
            args.baseline_run_dir,
            model=baseline_model,
            cfg=baseline_cfg,
            object_override=object_params,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
        )

        rows.append(
            {
                "object": object_name,
                "generic_success_rate": generic_metrics["success_rate"],
                "generic_mean_return": generic_metrics["mean_return"],
                "generic_std_return": generic_metrics["std_return"],
                "generic_n_episodes": generic_metrics["n_episodes"],
                "object_aware_success_rate": aware_metrics["success_rate"],
                "object_aware_mean_return": aware_metrics["mean_return"],
                "object_aware_std_return": aware_metrics["std_return"],
                "object_aware_n_episodes": aware_metrics["n_episodes"],
                "mass_g": object_params.mass_g,
                "rest_width_mm": object_params.rest_width_mm,
            }
        )

    if not rows:
        raise SystemExit(f"No per-object runs with env.object set found under {args.runs_dir}")

    rows.sort(key=lambda r: r["mass_g"])

    header = (
        f"{'object':<16} {'generic_success':>15} {'aware_success':>14} "
        f"{'generic_return':>15} {'aware_return':>13} {'mass_g':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['object']:<16} {r['generic_success_rate']:>15.2f} "
            f"{r['object_aware_success_rate']:>14.2f} {r['generic_mean_return']:>15.2f} "
            f"{r['object_aware_mean_return']:>13.2f} {r['mass_g']:>8.1f}"
        )

    csv_path = args.results_dir / "policy_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in _CSV_FIELDS})
    print(f"\nWrote {csv_path}")

    if args.plot:
        from plot_rollout_results import plot_comparison

        plot_comparison(rows, path=args.results_dir / "policy_comparison.png")


if __name__ == "__main__":
    main()
