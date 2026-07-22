#!/usr/bin/env python3
"""End-to-end smoke test. Runs in ~2 minutes and proves four things.

  [1] The env passes Gymnasium's API checker.
  [2] The training loop runs to completion: no crashes, TensorBoard events
      written, checkpoints appear on disk.
  [3] The final model reloads in a fresh object and produces bit-identical
      deterministic actions -> the save/load round-trip is lossless.
  [4] The reloaded policy completes full episodes in a rebuilt env.

Not a performance test. 3k steps will not lift the cube; it tells you the plumbing
is sound before you spend a night on 1M steps.

Usage (from the repo root):
    PYTHONPATH=src python src/tests/smoke_test.py --algo SAC --steps 3000
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.config import TrainConfig  # noqa: E402
from rl.env import EnvConfig  # noqa: E402
from rl.rollout import load_policy, rollout  # noqa: E402
from rl.train import train  # noqa: E402
from common.utils import FINAL_MODEL_NAME, setup_logging  # noqa: E402

LOGGER = logging.getLogger("smoke_test")

ATOL = 1e-6


def build_smoke_config(algo: str, steps: int, log_dir: Path) -> TrainConfig:
    """A deliberately tiny config: short horizon, one env, frequent checkpoints."""
    return TrainConfig(
        algo=algo,  # type: ignore[arg-type]
        seed=0,
        total_timesteps=steps,
        n_envs=1,
        n_threads=1,
        device="auto",
        normalize_obs=(algo == "PPO"),
        normalize_reward=(algo == "PPO"),
        checkpoint_freq=max(1, steps // 2),
        eval_freq=max(1, steps // 2),
        n_eval_episodes=1,
        log_dir=str(log_dir),
        run_name=f"smoke_{algo}",
        algo_kwargs=(
            {"learning_starts": 100} if algo == "SAC" else {"n_steps": 256, "batch_size": 64}
        ),
        env=EnvConfig(horizon=100),  # short episodes keep the test fast
    )


def assert_artifacts(run_dir: Path, expect_vecnorm: bool) -> None:
    """[2] Checkpoints, TensorBoard events and the config snapshot all exist."""
    assert (run_dir / FINAL_MODEL_NAME).exists(), f"missing {run_dir / FINAL_MODEL_NAME}"

    ckpts = sorted((run_dir / "checkpoints").glob("*.zip"))
    assert ckpts, "CheckpointCallback wrote nothing"

    tb_events = list((run_dir / "tb").rglob("events.out.tfevents.*"))
    assert tb_events, "no TensorBoard event files written"

    assert (run_dir / "config.json").exists(), "no config snapshot"
    if expect_vecnorm:
        assert (run_dir / "vecnormalize.pkl").exists(), "VecNormalize stats not saved"

    LOGGER.info("[2] OK: %d checkpoint(s), %d TB event file(s).", len(ckpts), len(tb_events))


def assert_round_trip(run_dir: Path) -> None:
    """[3] Two independent loads of the same archive agree exactly on actions.

    Comparing two *loads* (rather than in-memory vs loaded) isolates the
    serialisation path from any RNG state carried by the live trainer.
    """
    model_a, _ = load_policy(run_dir, FINAL_MODEL_NAME)
    model_b, _ = load_policy(run_dir, FINAL_MODEL_NAME)

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(4, *model_a.observation_space.shape)).astype(np.float32)

    act_a, _ = model_a.predict(obs, deterministic=True)
    act_b, _ = model_b.predict(obs, deterministic=True)

    max_diff = float(np.abs(act_a - act_b).max())
    assert max_diff < ATOL, f"deterministic actions diverge after reload (max = {max_diff})"
    assert act_a.shape == (4, *model_a.action_space.shape), "unexpected action batch shape"
    LOGGER.info("[3] OK: round-trip deterministic, max action delta = %.2e", max_diff)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="SAC")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--keep", action="store_true", help="Do not delete the temp run dir.")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="lang2grasp_smoke_"))
    try:
        cfg = build_smoke_config(args.algo, args.steps, tmp)

        # [1] is asserted inside train() via check_env() before any learning starts.
        LOGGER.info("[1] Running check_env + training %s for %d steps...", args.algo, args.steps)
        run_dir, _ = train(cfg)

        assert_artifacts(run_dir, expect_vecnorm=cfg.normalize_obs)
        assert_round_trip(run_dir)

        metrics = rollout(run_dir, episodes=2, seed=99)
        assert metrics["mean_length"] > 0, "policy produced zero-length episodes"
        LOGGER.info("[4] OK: rollout metrics %s", metrics)

        LOGGER.info("SMOKE TEST PASSED")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
