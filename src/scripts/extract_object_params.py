"""Stage 1 (+ 1.5): text prompt -> LLM -> object physical parameters, snapshotted to
JSON, immediately followed by an accuracy check against the golden dataset.

Decouples training from live LLM calls: run this once against a real backend, and
every later training run -- including on a SLURM node with no internet -- reads the
resulting `configs/objects/<name>.json` snapshot instead of calling an LLM again.
These snapshots are gitignored (LLM output, regenerate on demand), so run this
before stage 2/3 on any fresh checkout.

Right after writing the snapshots, this also runs `evaluate_extraction_accuracy.py`'s
comparison against the golden dataset -- skipped (with a log message, not an error) if
none of the extracted names are in `extraction.param_prompts.PRIORS`, e.g. a custom
`--prompts` file for objects with no golden entry. Pass `--no-evaluate` to skip it
unconditionally, e.g. when scripting many extraction runs back-to-back.

`--samples N` (default 1) draws N independent extractions per object and combines
them (median per numeric field, majority vote per categorical field) instead of
trusting a single call -- even at temperature 0, a single call has real sample-to-
sample noise on the harder-to-calibrate fields (grip/crush force especially; see
`extraction.param_extraction`'s module docstring). Costs N backend calls per object.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend anthropic --model claude-haiku-4-5
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend openai --model gpt-4o-mini
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend groq --model openai/gpt-oss-120b --plot
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend groq --samples 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from evaluate_extraction_accuracy import evaluate
from extraction.llm_backends import BACKENDS
from extraction.param_extraction import extract_object_params
from extraction.param_prompts import PRIORS
from common.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--model", default=None, help="Override the backend's default model.")
    parser.add_argument("--prompts", type=Path, default=Path("src/configs/objects/prompts.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("src/configs/objects"))
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Independent extractions per object, combined via median/majority vote (default: 1).",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("src/results"), help="Stage 1.5 output dir.")
    parser.add_argument("--plot", action="store_true", help="Also save stage 1.5's accuracy chart.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip the stage 1.5 accuracy check.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    raw = json.loads(args.prompts.read_text())
    objects: dict[str, str] = raw["objects"]

    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(**({"model": args.model} if args.model else {}))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in objects.items():
        params = extract_object_params(name, prompt, backend, samples=args.samples)
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(dataclasses.asdict(params), indent=2))
        LOGGER.info("%s -> %s | %s", name, out_path, params)

    LOGGER.info("Wrote %d object snapshot(s) to %s", len(objects), args.out_dir)

    if args.no_evaluate:
        return
    if not (set(objects) & set(PRIORS)):
        LOGGER.info("None of %s are in PRIORS' golden dataset; skipping stage 1.5.", sorted(objects))
        return
    print()
    evaluate(args.out_dir, args.results_dir, args.plot)


if __name__ == "__main__":
    main()
