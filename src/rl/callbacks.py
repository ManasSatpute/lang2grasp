"""SB3 callbacks: checkpointing, periodic eval, graceful pre-emption."""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from types import FrameType

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecEnv

from rl.config import TrainConfig
from common.utils import CHECKPOINT_PREFIX

LOGGER = logging.getLogger(__name__)


class GracefulExitCallback(BaseCallback):
    """Stop training cleanly on a wall-clock budget or a SLURM pre-emption signal.

    SLURM kills a job at its ``--time`` limit with no warning unless you ask for one
    via ``--signal=B:USR1@300``. Without this callback the last N minutes of compute
    are lost and the run resumes from the previous checkpoint.

    Returning ``False`` from ``_on_step`` makes ``learn()`` unwind normally, so
    ``train()``'s ``finally`` block runs and writes the model, VecNormalize
    statistics and replay buffer.
    """

    def __init__(self, max_hours: float | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._deadline = time.monotonic() + max_hours * 3600 if max_hours else None
        self._signalled = False
        self.stop_reason: str | None = None

    def install_signal_handlers(self) -> None:
        """Trap SIGUSR1 (pre-emption warning) and SIGTERM (scancel / time limit)."""
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        # Signal handlers must be fast and re-entrant: set a flag, nothing more.
        self._signalled = True
        LOGGER.warning("Caught signal %s; will checkpoint and exit at the next step.", signum)

    def _on_step(self) -> bool:
        if self._signalled:
            self.stop_reason = "signal"
            return False
        if self._deadline is not None and time.monotonic() > self._deadline:
            self.stop_reason = "wallclock"
            LOGGER.warning("Wall-clock budget exhausted; checkpointing and exiting.")
            return False
        return True

    @property
    def interrupted(self) -> bool:
        return self.stop_reason is not None


def build_callbacks(
    cfg: TrainConfig, run_dir: Path, eval_env: VecEnv
) -> tuple[CallbackList, GracefulExitCallback]:
    """Return the callback stack plus a handle on the pre-emption callback.

    Callback frequencies in SB3 count *policy* steps, not environment steps, so we
    divide by ``n_envs`` to keep the wall-clock cadence stable when scaling parallelism.
    """
    per_env = max(1, cfg.n_envs)
    graceful = GracefulExitCallback(max_hours=cfg.max_hours)
    graceful.install_signal_handlers()

    callbacks: list[BaseCallback] = [
        graceful,
        CheckpointCallback(
            save_freq=max(1, cfg.checkpoint_freq // per_env),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=CHECKPOINT_PREFIX,
            # The replay buffer is written once on exit instead: pickling ~200 MB
            # every 50k steps would dominate the job's I/O on a shared filesystem.
            save_replay_buffer=False,
            save_vecnormalize=cfg.normalize_obs or cfg.normalize_reward,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "eval"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(1, cfg.eval_freq // per_env),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
    ]

    LOGGER.info("Callbacks: %s", [type(c).__name__ for c in callbacks])
    return CallbackList(callbacks), graceful
