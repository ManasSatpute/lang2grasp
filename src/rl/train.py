"""Train a stock SB3 SAC agent on robosuite Lift (Panda).

No custom policies -- the point is a clean, verifiable baseline. Optionally adds
grip-force-aware reward shaping (see `rl.env.EnvConfig.grip_force_shaping`), but
that's config-gated and off by default. Survives SLURM wall-time limits:
checkpoints on SIGUSR1, exits with EXIT_REQUEUE, and `--resume` picks up from the
newest checkpoint.

Usage (from the repo root):
    PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json
    PYTHONPATH=src python -m rl.train --config src/configs/policy/sac.json --total-timesteps 200000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rl.callbacks import build_callbacks
from rl.config import TrainConfig, load_config, save_config
from rl.env import make_lift_env
from common.utils import (
    CONFIG_SNAPSHOT,
    EXIT_REQUEUE,
    FINAL_MODEL_NAME,
    REPLAY_BUFFER_NAME,
    VECNORM_NAME,
    configure_threads,
    cpu_budget,
    find_latest_checkpoint,
    make_run_dir,
    resolve_device,
    set_global_seed,
    setup_logging,
)
from rl.vec_env import build_vec_env

LOGGER = logging.getLogger(__name__)

#: Constant across resumes so TensorBoard keeps appending to one event stream
#: instead of creating tb_1, tb_2, ... on every requeue.
TB_LOG_NAME = "tb"


def _validate_env_once(cfg: TrainConfig) -> None:
    """Run Gymnasium's API conformance checker on a single un-vectorised env.

    Catches space/dtype mismatches and reset-signature bugs *before* burning hours
    of queue time on a run that would have crashed at step 500.
    """
    env = make_lift_env(cfg.env)
    try:
        check_env(env, warn=True, skip_render_check=True)
        LOGGER.info("check_env passed.")
    finally:
        env.close()


def _warn_on_oversubscription(cfg: TrainConfig) -> None:
    """12 env workers + a learner on a 12-core slot will contend for cores."""
    budget = cpu_budget()
    requested = cfg.n_envs + cfg.n_threads
    if requested > budget:
        LOGGER.warning(
            "n_envs(%d) + n_threads(%d) = %d exceeds the %d-core allocation. "
            "Expect thread thrash; lower n_envs.",
            cfg.n_envs,
            cfg.n_threads,
            requested,
            budget,
        )


def _build_model(
    cfg: TrainConfig, train_env: VecEnv, run_dir: Path, device: object
) -> BaseAlgorithm:
    """Construct a fresh model, or reload the newest checkpoint when resuming."""
    ckpt = find_latest_checkpoint(run_dir) if cfg.resume else None

    if ckpt is None:
        if cfg.resume:
            LOGGER.info("--resume given but no checkpoint found; starting fresh.")
        return SAC(
            "MlpPolicy",
            train_env,
            seed=cfg.seed,
            device=device,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=cfg.policy_kwargs or None,
            **cfg.algo_kwargs,
        )

    LOGGER.info("Resuming from %s", ckpt)
    model = SAC.load(ckpt, env=train_env, device=device, tensorboard_log=str(run_dir / "tb"))

    # SAC without its replay buffer restarts with an empty buffer: the critic
    # unlearns, and the return curve visibly craters at every requeue boundary.
    buffer = run_dir / REPLAY_BUFFER_NAME
    if buffer.exists():
        model.load_replay_buffer(buffer)
        LOGGER.info("Restored replay buffer (%d transitions).", model.replay_buffer.size())
    else:
        LOGGER.warning("No replay buffer at %s; SAC restarts cold from here.", buffer)

    LOGGER.info("Resumed at %d / %d timesteps.", model.num_timesteps, cfg.total_timesteps)
    return model


def train(cfg: TrainConfig) -> tuple[Path, bool]:
    """Run one training job. Returns ``(run_dir, needs_requeue)``."""
    configure_threads(cfg.n_threads)
    set_global_seed(cfg.seed)
    _warn_on_oversubscription(cfg)
    device = resolve_device(cfg.device)

    run_name = cfg.run_name or "SAC_local"
    run_dir = make_run_dir(cfg.log_dir, run_name)
    save_config(cfg, run_dir / CONFIG_SNAPSHOT)  # no-op if resuming
    LOGGER.info("Run directory: %s | device: %s", run_dir, device)

    _validate_env_once(cfg)

    train_env = build_vec_env(cfg, n_envs=cfg.n_envs, seed=cfg.seed, training=True)
    # Offset the eval seed so evaluation never replays the training initial states.
    eval_env = build_vec_env(cfg, n_envs=1, seed=cfg.seed + 10_000, training=False)
    if isinstance(train_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        eval_env.obs_rms = train_env.obs_rms  # eval must use the training statistics

    model = _build_model(cfg, train_env, run_dir, device)
    callbacks, graceful = build_callbacks(cfg, run_dir, eval_env)

    resuming = model.num_timesteps > 0
    # SB3 gotcha: with reset_num_timesteps=False, `_setup_learn` does
    # `total_timesteps += self.num_timesteps`. Passing the global budget on a resume
    # would train for budget + already_done steps. Pass the *remaining* budget.
    remaining = cfg.total_timesteps - model.num_timesteps
    if remaining <= 0:
        LOGGER.info("Budget already met (%d steps). Nothing to do.", model.num_timesteps)
        train_env.close()
        eval_env.close()
        return run_dir, False

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=not resuming,
            tb_log_name=TB_LOG_NAME,
            progress_bar=False,  # a tqdm bar in a batch .out file is unreadable noise
        )
    finally:
        # Always persist, even on signal or exception: a partial run beats no run.
        model.save(run_dir / FINAL_MODEL_NAME)
        if cfg.save_replay_buffer:
            model.save_replay_buffer(run_dir / REPLAY_BUFFER_NAME)
        if isinstance(train_env, VecNormalize):
            train_env.save(str(run_dir / VECNORM_NAME))
        train_env.close()
        eval_env.close()

    finished = model.num_timesteps >= cfg.total_timesteps
    needs_requeue = graceful.interrupted and not finished
    LOGGER.info(
        "Stopped at %d/%d steps (reason=%s). Requeue needed: %s",
        model.num_timesteps,
        cfg.total_timesteps,
        graceful.stop_reason or "budget reached",
        needs_requeue,
    )
    return run_dir, needs_requeue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON config.")
    parser.add_argument("--total-timesteps", dest="total_timesteps", type=int, default=None)
    parser.add_argument("--n-envs", dest="n_envs", type=int, default=None)
    parser.add_argument("--n-threads", dest="n_threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--max-hours", dest="max_hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config, overrides=vars(args) | {"config": None})
    _, needs_requeue = train(cfg)
    # slurm/train.slurm reads this exit code to decide whether to requeue.
    sys.exit(EXIT_REQUEUE if needs_requeue else 0)


if __name__ == "__main__":
    main()
