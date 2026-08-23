"""Phase 2, the paradigm switch: train pi_blind / pi_blind+hist / pi_param against a
*continuous distribution* of objects, instead of one per-object specialist.

All three variants share the same domain-randomized env (``EnvConfig.
randomize_object``): a fresh ``ObjectParams`` is sampled every episode. They differ
only in what the policy sees: ``blind`` sees nothing extra (the memoryless floor);
``blind_hist`` adds a GRU (``rl.policies.HistoryGRUExtractor``) over recent
proprioception + force for implicit system identification; ``param`` FiLM-conditions
on a (possibly noisy) object-parameter vector z (``rl.policies.FiLMExtractor``).
Per-object specialists (``scripts/train_object.py``) are a separate oracle topline,
untouched by this script.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind \\
        --base-config src/configs/policy/sac.json
    PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind_hist \\
        --base-config src/configs/policy/sac.json --history-len 16
    PYTHONPATH=src python src/scripts/train_paradigm.py --variant param \\
        --base-config src/configs/policy/sac.json --z-noise-std 0.1
    PYTHONPATH=src python src/scripts/train_paradigm.py --variant blind \\
        --total-timesteps 2000   # quick smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rl.config import TRAIN_OVERRIDE_FIELDS, TrainConfig, add_override_args, load_config
from objects.object_params import Z_DIM
from rl.env import history_dims
from rl.policies import FiLMExtractor, HistoryGRUExtractor
from rl.train import train
from common.utils import EXIT_REQUEUE, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["blind", "blind_hist", "param"], required=True,
        help="Which paradigm-switch policy to train (see module docstring).",
    )
    parser.add_argument("--base-config", type=Path, default=Path("src/configs/policy/sac.json"))
    add_override_args(parser)
    parser.add_argument(
        "--shapes", nargs="+", choices=["box", "cylinder", "ball"], default=None,
        help="Restrict domain randomization to this subset of shapes (default: all three).",
    )
    parser.add_argument(
        "--history-len", type=int, default=16,
        help="GRU history window length (--variant blind_hist only).",
    )
    parser.add_argument(
        "--gru-hidden", type=int, default=128,
        help="HistoryGRUExtractor's GRU hidden size (--variant blind_hist only).",
    )
    parser.add_argument(
        "--z-noise-std", type=float, default=0.1,
        help="Relative Gaussian noise on z's continuous dims (--variant param only). "
        "0.0 = exact/noiseless z.",
    )
    parser.add_argument(
        "--grip-force-shaping", action="store_true",
        help="Enable EnvConfig.grip_force_shaping (off by default) -- see "
        "scripts/train_object.py's --grip-force-shaping for what this does.",
    )
    return parser.parse_args()


def _configure_variant(cfg: TrainConfig, args: argparse.Namespace) -> None:
    """Mutate ``cfg`` in place: env randomization + the variant's obs/feature wiring."""
    cfg.env.randomize_object = True
    if args.shapes:
        cfg.env.randomize_shapes = tuple(args.shapes)
    cfg.env.grip_force_shaping = args.grip_force_shaping

    if args.variant == "blind":
        return  # stock MlpPolicy, no history, no z -- the memoryless floor

    if args.variant == "blind_hist":
        cfg.env.history_len = args.history_len
        current_dim, history_len, step_dim = history_dims(cfg.env)
        cfg.policy_kwargs = {
            **cfg.policy_kwargs,
            "features_extractor_class": HistoryGRUExtractor,
            "features_extractor_kwargs": {
                "current_dim": current_dim,
                "history_len": history_len,
                "step_dim": step_dim,
                "gru_hidden": args.gru_hidden,
            },
        }
        return

    if args.variant == "param":
        cfg.env.include_object_z = True
        cfg.env.object_z_noise_std = args.z_noise_std
        cfg.policy_kwargs = {
            **cfg.policy_kwargs,
            "features_extractor_class": FiLMExtractor,
            "features_extractor_kwargs": {"z_dim": Z_DIM},
        }
        return

    raise ValueError(f"Unknown variant: {args.variant!r}")  # unreachable given argparse choices


def main() -> None:
    setup_logging()
    args = parse_args()

    overrides = {key: getattr(args, key) for key in TRAIN_OVERRIDE_FIELDS}
    cfg = load_config(args.base_config, overrides=overrides)
    _configure_variant(cfg, args)
    if cfg.run_name is None:
        cfg.run_name = f"paradigm_{args.variant}"

    _, needs_requeue = train(cfg)
    sys.exit(EXIT_REQUEUE if needs_requeue else 0)


if __name__ == "__main__":
    main()
