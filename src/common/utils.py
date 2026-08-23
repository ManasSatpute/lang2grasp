"""Small shared helpers: logging, seeding, device, threads, run directories."""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "model"
FINAL_MODEL_NAME = "final_model.zip"
VECNORM_NAME = "vecnormalize.pkl"
REPLAY_BUFFER_NAME = "replay_buffer.pkl"
CONFIG_SNAPSHOT = "config.json"
#: The resolved composite controller config robosuite actually ran with (see rl.env.resolve_controller_config).
CONTROLLER_CONFIG_SNAPSHOT = "controller_config.json"

#: Exit code meaning "checkpointed cleanly, work remains -- please requeue" (see src/slurm/train.slurm).
EXIT_REQUEUE = 42


def setup_logging(level: int = logging.INFO) -> None:
    """One-shot root logger config. Idempotent so scripts can call it freely."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # robosuite logs to "robosuite_logs" AND propagates to root, so lines print twice
    # unless both loggers are silenced.
    for name in ("robosuite", "robosuite_logs"):
        logging.getLogger(name).setLevel(logging.ERROR)


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch. robosuite additionally needs the *global* numpy RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_budget() -> int:
    """How many cores this process may actually use (not the node's full count under SLURM)."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    try:
        return len(os.sched_getaffinity(0))  # respects cgroup/taskset pinning
    except AttributeError:  # pragma: no cover - macOS/Windows
        return os.cpu_count() or 1


def configure_threads(n_threads: int = 1) -> None:
    """Pin torch thread counts (BLAS threads are set separately via OMP_NUM_THREADS)."""
    n_threads = max(1, n_threads)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)
    LOGGER.info("torch threads=%d (cpu budget=%d)", n_threads, cpu_budget())


def resolve_device(requested: str = "auto") -> torch.device:
    """Map ``"auto"`` to cuda-if-present, and warn on an impossible request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Requested %s but CUDA is unavailable; falling back to CPU.", requested)
        return torch.device("cpu")
    return device


def make_run_dir(log_dir: str | Path, run_name: str) -> Path:
    """Create (or reuse) the directory holding one run's artifacts."""
    run_dir = Path(log_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the highest-step checkpoint in ``run_dir/checkpoints`` (numeric, not lexical, order), or None."""
    ckpts = list(Path(run_dir).glob(f"checkpoints/{CHECKPOINT_PREFIX}_*_steps.zip"))
    scored = [(int(m.group(1)), p) for p in ckpts if (m := _STEP_RE.search(p.name))]
    if not scored:
        return None
    return max(scored)[1]
