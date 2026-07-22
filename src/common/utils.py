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

#: Exit code meaning "I checkpointed cleanly and there is work left; please requeue."
#: src/slurm/train.slurm keys off this. Anything else is a genuine success/failure.
EXIT_REQUEUE = 42


def setup_logging(level: int = logging.INFO) -> None:
    """One-shot root logger config. Idempotent so scripts can call it freely."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # robosuite logs to a logger named "robosuite_logs" that has its OWN handler and
    # also propagates to root -- hence every line appearing twice. Raising its level
    # silences both copies. The warnings it emits for Panda are all benign: the
    # composite BASIC controller config covers every body part of every robot, and a
    # Panda simply has no torso/head/legs/base/left-arm to configure.
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
    """How many cores this process may actually use.

    Under SLURM, ``os.cpu_count()`` reports the *node's* core count, not the 12 you
    were allocated. Oversubscribing a shared node makes your job slower, not faster.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    try:
        return len(os.sched_getaffinity(0))  # respects cgroup/taskset pinning
    except AttributeError:  # pragma: no cover - macOS/Windows
        return os.cpu_count() or 1


def configure_threads(n_threads: int = 1) -> None:
    """Pin torch thread counts for the learner process.

    The BLAS side is set via OMP_NUM_THREADS in the .slurm scripts, because BLAS
    reads it at import and every SubprocVecEnv worker links BLAS independently.
    """
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
    """Create (or reuse) the directory holding one run's artifacts.

    The name must be stable across requeues, so the .slurm script derives it from
    $SLURM_JOB_ID. A timestamped name would silently start from scratch on restart.
    """
    run_dir = Path(log_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    return run_dir


_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the highest-step checkpoint in ``run_dir/checkpoints``, or None.

    Sorted numerically, not lexically: "950000" > "1000000" as strings, which would
    resume the same checkpoint forever.
    """
    ckpts = list(Path(run_dir).glob(f"checkpoints/{CHECKPOINT_PREFIX}_*_steps.zip"))
    scored = [(int(m.group(1)), p) for p in ckpts if (m := _STEP_RE.search(p.name))]
    if not scored:
        return None
    return max(scored)[1]
