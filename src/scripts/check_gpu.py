#!/usr/bin/env python3
"""Verify the compute stack and log a 'hello world' tensor from the GPU.

Run this as a batch job on the same partition you will train on. A login node has
no GPU, so checking there proves nothing.

Note: MuJoCo physics is CPU-only. The GPU accelerates the policy network and
offscreen rendering, nothing else.

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/check_gpu.py
"""

from __future__ import annotations

import logging
import os
import platform
import sys

# Fallback for running this file directly (without PYTHONPATH=src set): add the
# src/ dir (this file's parent) so common/rl/objects/extraction resolve as top-level packages.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from common.utils import cpu_budget, setup_logging  # noqa: E402

LOGGER = logging.getLogger("check_gpu")


def log_torch_and_gpu() -> None:
    """Log device details and prove a kernel launches by materialising a tensor."""
    LOGGER.info("python      : %s (%s)", platform.python_version(), platform.machine())
    LOGGER.info("torch       : %s", torch.__version__)
    LOGGER.info("torch cuda  : %s", torch.version.cuda or "not built with CUDA")
    LOGGER.info("cuda avail  : %s", torch.cuda.is_available())
    LOGGER.info("cpu budget  : %d cores", cpu_budget())

    if not torch.cuda.is_available():
        LOGGER.warning("No CUDA device visible. Did the job request a GPU?")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        LOGGER.info(
            "device 0    : %s (sm_%d%d, %.1f GiB)",
            props.name,
            props.major,
            props.minor,
            props.total_memory / 1024**3,
        )

    # The actual "hello world": allocate, compute, read back. This catches a
    # driver/toolkit mismatch that `is_available()` happily reports as True.
    x = torch.ones(3, 3, device=device)
    y = (x @ x).sum().item()
    LOGGER.info("hello world from %s | ones(3,3) @ ones(3,3) sum = %.1f (expected 27.0)", device, y)
    assert abs(y - 27.0) < 1e-6, "matmul returned the wrong result -- broken CUDA install"


def log_rl_stack() -> None:
    """Import and version-log the simulation + RL dependencies."""
    import gymnasium
    import mujoco
    import numpy as np
    import robosuite
    import stable_baselines3 as sb3

    LOGGER.info("numpy       : %s", np.__version__)
    LOGGER.info("gymnasium   : %s", gymnasium.__version__)
    LOGGER.info("mujoco      : %s", mujoco.__version__)
    LOGGER.info("robosuite   : %s", robosuite.__version__)
    LOGGER.info("sb3         : %s", sb3.__version__)
    LOGGER.info("MUJOCO_GL   : %s", os.environ.get("MUJOCO_GL", "<unset>"))


def main() -> int:
    setup_logging()
    LOGGER.info("=" * 70)
    log_torch_and_gpu()
    LOGGER.info("-" * 70)
    log_rl_stack()
    LOGGER.info("=" * 70)
    LOGGER.info("Stack check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
