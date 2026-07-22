"""Vectorised environment construction, shared by training and evaluation.

Kept in one place so that ``rollout.py`` rebuilds *exactly* the observation
pipeline the policy was trained on. A mismatched ``VecNormalize`` is the single
most common cause of "my saved policy performs at chance".
"""

from __future__ import annotations

import logging
from pathlib import Path

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from rl.config import TrainConfig
from rl.env import EnvConfig, make_lift_env

LOGGER = logging.getLogger(__name__)


def build_vec_env(
    cfg: TrainConfig,
    n_envs: int,
    seed: int,
    *,
    training: bool,
    env_cfg: EnvConfig | None = None,
    vecnormalize_path: str | Path | None = None,
) -> VecEnv:
    """Build a (optionally normalised) vectorised robosuite env.

    Args:
        cfg: Resolved training config.
        n_envs: Number of parallel workers. >1 uses subprocesses.
        seed: Base seed; worker ``i`` gets ``seed + i``.
        training: If False, ``VecNormalize`` freezes its running statistics and
            stops normalising the reward, so logged returns stay interpretable.
        env_cfg: Override the env config (e.g. to enable ``rgb_array`` rendering).
        vecnormalize_path: Load pre-fitted normalisation statistics from disk.
    """
    env_cfg = env_cfg or cfg.env
    use_subproc = n_envs > 1

    venv = make_vec_env(
        make_lift_env,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"cfg": env_cfg},
        vec_env_cls=SubprocVecEnv if use_subproc else DummyVecEnv,
        # MuJoCo contexts do not survive fork(); spawn is the safe start method.
        vec_env_kwargs={"start_method": "spawn"} if use_subproc else None,
    )

    if vecnormalize_path is not None:
        venv = VecNormalize.load(str(vecnormalize_path), venv)
        venv.training = training
        venv.norm_reward = training and cfg.normalize_reward
        LOGGER.info("Loaded VecNormalize stats from %s (training=%s)", vecnormalize_path, training)
    elif cfg.normalize_obs or cfg.normalize_reward:
        venv = VecNormalize(
            venv,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward and training,
            training=training,
            clip_obs=10.0,
        )

    return venv
