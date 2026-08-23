"""Stage 1.5: how accurate is LLM extraction against the golden (ground-truth) dataset?

Compares `extract_object_params.py`'s per-object snapshots against
`extraction.param_prompts.PRIORS` (the ground-truth values for the same 6 objects):
categorical fields (`shape`, `mass_class`, `fragile`) as a match rate; numeric fields
as mean absolute and relative error.

`extract_object_params.py` runs this automatically right after extraction (stage 1 +
1.5 as one step) whenever the extracted names overlap `PRIORS`. Run this script
directly only to re-check/re-plot existing snapshots without calling an LLM again:

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/evaluate_extraction_accuracy.py --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from extraction.param_prompts import PRIORS, golden_object_params
from objects.object_params import ObjectParams
from common.utils import setup_logging

LOGGER = logging.getLogger(__name__)

#: (field label, extractor) -- includes derived properties (mass_g, rest_width_mm)
#: alongside plain ObjectParams attributes, compared uniformly via the same
#: abs/pct-error machinery.
_NUMERIC_FIELDS: tuple[tuple[str, Callable[[ObjectParams], float]], ...] = (
    ("density_kg_m3", lambda p: p.density),
    ("friction_sliding", lambda p: p.friction[0]),
    ("friction_torsional", lambda p: p.friction[1]),
    ("friction_rolling", lambda p: p.friction[2]),
    ("grip_force_min_N", lambda p: p.grip_force_min_N),
    ("grip_force_max_N", lambda p: p.grip_force_max_N),
    ("spring_Npm", lambda p: p.spring_Npm),
    ("crush_force_N", lambda p: p.crush_force_N),
    ("mass_g", lambda p: p.mass_g),
    ("rest_width_mm", lambda p: p.rest_width_mm),
)
_CATEGORICAL_FIELDS: tuple[str, ...] = ("shape", "mass_class", "fragile")

#: Below this, a relative/percent error is meaningless (near-zero denominator).
#: None of PRIORS' shipped values are ever this small, but an LLM-extracted golden
#: substitute in principle could be -- guard rather than divide by ~0.
_MIN_DENOM = 1e-6

_DETAIL_CSV_FIELDS = ("object", "field", "kind", "golden", "extracted", "abs_error", "pct_error", "match")
_SUMMARY_CSV_FIELDS = ("field", "kind", "n", "mean_abs_error", "mean_pct_error", "accuracy")


def _load_extracted(objects_dir: Path, name: str) -> ObjectParams | None:
    path = objects_dir / f"{name}.json"
    if not path.exists():
        return None
    return ObjectParams(**json.loads(path.read_text()))


def compare_object(golden: ObjectParams, extracted: ObjectParams) -> list[dict]:
    """Per-field rows for one object: numeric (abs/pct error) and categorical (match)."""
    rows = []
    for field, get in _NUMERIC_FIELDS:
        g, e = get(golden), get(extracted)
        abs_error = abs(e - g)
        pct_error = 100.0 * abs_error / abs(g) if abs(g) > _MIN_DENOM else None
        rows.append(
            {
                "object": golden.name,
                "field": field,
                "kind": "numeric",
                "golden": g,
                "extracted": e,
                "abs_error": abs_error,
                "pct_error": pct_error,
                "match": None,
            }
        )
    for field in _CATEGORICAL_FIELDS:
        g, e = getattr(golden, field), getattr(extracted, field)
        rows.append(
            {
                "object": golden.name,
                "field": field,
                "kind": "categorical",
                "golden": g,
                "extracted": e,
                "abs_error": None,
                "pct_error": None,
                "match": float(g == e),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """One row per (field, kind): n, mean abs/pct error (numeric) or accuracy (categorical)."""
    by_field: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_field[(row["field"], row["kind"])].append(row)

    summary = []
    for (field, kind), field_rows in by_field.items():
        n = len(field_rows)
        if kind == "numeric":
            pct_errors = [r["pct_error"] for r in field_rows if r["pct_error"] is not None]
            summary.append(
                {
                    "field": field,
                    "kind": kind,
                    "n": n,
                    "mean_abs_error": sum(r["abs_error"] for r in field_rows) / n,
                    "mean_pct_error": (sum(pct_errors) / len(pct_errors)) if pct_errors else None,
                    "accuracy": None,
                }
            )
        else:
            summary.append(
                {
                    "field": field,
                    "kind": kind,
                    "n": n,
                    "mean_abs_error": None,
                    "mean_pct_error": None,
                    "accuracy": 100.0 * sum(r["match"] for r in field_rows) / n,
                }
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("src/configs/objects"),
        help="Directory holding extract_object_params.py's <name>.json snapshots.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("src/results"))
    parser.add_argument("--plot", action="store_true", help="Save an accuracy chart.")
    return parser.parse_args()


def evaluate(extracted_dir: Path, results_dir: Path, plot: bool = False) -> None:
    """Compare every snapshot in ``extracted_dir`` against the golden dataset, print a
    report, and write ``results_dir``'s CSVs (and, if ``plot``, its chart). Also callable
    directly from `extract_object_params.py` so extraction + accuracy run as one step.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    skipped: list[str] = []
    for name in sorted(PRIORS):
        golden = golden_object_params(name)
        extracted = _load_extracted(extracted_dir, name)
        if extracted is None:
            skipped.append(name)
            continue
        all_rows.extend(compare_object(golden, extracted))

    if not all_rows:
        raise SystemExit(
            f"No extracted snapshots found in {extracted_dir} for any of PRIORS' "
            f"{len(PRIORS)} named objects. Run scripts/extract_object_params.py first."
        )
    if skipped:
        LOGGER.warning("No extracted snapshot for: %s (skipped).", skipped)

    summary = summarize(all_rows)
    numeric_summary = [r for r in summary if r["kind"] == "numeric"]
    numeric_summary.sort(key=lambda r: r["mean_pct_error"] or -1.0, reverse=True)
    categorical_summary = [r for r in summary if r["kind"] == "categorical"]
    categorical_summary.sort(key=lambda r: r["accuracy"])

    n_objects = len(PRIORS) - len(skipped)
    print(f"Extraction accuracy vs. golden dataset ({n_objects}/{len(PRIORS)} objects)\n")
    print(f"{'field':<20} {'mean abs error':>16} {'mean % error':>14} {'n':>4}")
    print("-" * 58)
    for r in numeric_summary:
        pct = f"{r['mean_pct_error']:.1f}%" if r["mean_pct_error"] is not None else "n/a"
        print(f"{r['field']:<20} {r['mean_abs_error']:>16.4g} {pct:>14} {r['n']:>4}")
    print(
        "\nNote: friction_torsional/rolling are tiny-magnitude coefficients "
        "(golden ~0.0001-0.01) -- a small absolute miss there can read as a large "
        "% error. Check mean abs error too, not just %."
    )

    print(f"\n{'field':<20} {'accuracy':>10} {'n':>4}")
    print("-" * 38)
    for r in categorical_summary:
        print(f"{r['field']:<20} {r['accuracy']:>9.1f}% {r['n']:>4}")

    detail_path = results_dir / "extraction_accuracy_detail.csv"
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DETAIL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {detail_path}")

    summary_path = results_dir / "extraction_accuracy_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {summary_path}")

    if plot:
        from plot_rollout_results import plot_extraction_accuracy

        plot_extraction_accuracy(numeric_summary, categorical_summary, path=results_dir / "extraction_accuracy.png")


def main() -> None:
    setup_logging()
    args = parse_args()
    evaluate(args.extracted_dir, args.results_dir, args.plot)


if __name__ == "__main__":
    main()
