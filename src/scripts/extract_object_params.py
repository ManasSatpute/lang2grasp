"""Stage 1: text prompt -> LLM -> object physical parameters, snapshotted to JSON.

Decouples training from live LLM calls: run this once (with a real backend or
offline `mock`), commit the resulting `configs/objects/<name>.json` files, and
every later training run -- including on a SLURM node with no internet -- reads
the snapshot instead of calling an LLM again.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend mock
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend anthropic --model claude-haiku-4-5
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend openai --model gpt-4o-mini
    PYTHONPATH=src python src/scripts/extract_object_params.py --backend groq --model llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from extraction.llm_backends import BACKENDS, MockBackend
from extraction.param_extraction import extract_object_params
from common.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="mock")
    parser.add_argument("--model", default=None, help="Override the backend's default model.")
    parser.add_argument("--prompts", type=Path, default=Path("src/configs/objects/prompts.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("src/configs/objects"))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    raw = json.loads(args.prompts.read_text())
    objects: dict[str, str] = raw["objects"]

    backend_cls = BACKENDS[args.backend]
    backend = MockBackend() if args.backend == "mock" else backend_cls(**({"model": args.model} if args.model else {}))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, prompt in objects.items():
        params = extract_object_params(name, prompt, backend)
        out_path = args.out_dir / f"{name}.json"
        out_path.write_text(json.dumps(dataclasses.asdict(params), indent=2))
        LOGGER.info("%s -> %s | %s", name, out_path, params)

    LOGGER.info("Wrote %d object snapshot(s) to %s", len(objects), args.out_dir)


if __name__ == "__main__":
    main()
