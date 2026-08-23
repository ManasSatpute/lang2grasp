"""Replay-buffer seeding for the sparse/"fixed" reward config (``EnvConfig.reward_shaping=False``).

With a sparse reward, a freshly-initialised SAC policy essentially never lifts the
cube by chance, so the replay buffer fills with all-zero-reward transitions and the
critic never sees a gradient. This module runs a scripted reach/descend/grasp/lift
heuristic (`rl/scripted_policy.ScriptedPickPolicy`) against a throwaway env to find a
handful of real successes and inserts their transitions into SAC's replay buffer
before training starts. Only called from `rl/train.py` when reward_shaping is False.
"""

from __future__ import annotations

import logging

import numpy as np
from stable_baselines3 import SAC

from rl.config import TrainConfig
from rl.env import RobosuiteLiftEnv
from rl.scripted_policy import MIN_ACTION_DIM, ScriptedPickPolicy

LOGGER = logging.getLogger(__name__)


def _rollout_scripted_episode(
    env: RobosuiteLiftEnv, rng: np.random.Generator
) -> tuple[list[tuple], bool]:
    """One scripted episode. Returns (transitions, succeeded)."""
    low, high = env.action_space.low, env.action_space.high
    if low.shape[0] < MIN_ACTION_DIM:
        return [], False

    policy = ScriptedPickPolicy(low, high, rng)
    obs, _ = env.reset()
    transitions: list[tuple] = []
    succeeded = False
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.act(env.last_obs_dict)
        if action is None:
            return [], False
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs, next_obs, action, reward, terminated, truncated))
        succeeded = succeeded or bool(info.get("is_success", False))
        obs = next_obs

    return transitions, succeeded


def seed_replay_buffer_with_successes(
    model: SAC,
    cfg: TrainConfig,
    *,
    target_successes: int = 20,
    max_attempts: int = 300,
) -> int:
    """Pre-fill ``model``'s replay buffer with real successful scripted episodes.

    Only meaningful for the sparse/"fixed" reward config -- callers must check
    ``cfg.env.reward_shaping is False`` themselves (see ``rl/train.py``). Returns the
    number of successful episodes actually added (0 if the heuristic never
    succeeded, or if seeding was skipped -- both are non-fatal; training proceeds
    with whatever the buffer already has).
    """
    if cfg.normalize_obs or cfg.normalize_reward:
        LOGGER.warning(
            "Skipping sparse-reward replay seeding: normalize_obs/normalize_reward "
            "would need VecNormalize statistics applied to the seeded transitions "
            "too, which isn't implemented here. Train with normalize_obs=false and "
            "normalize_reward=false (the default), or seed manually."
        )
        return 0

    rng = np.random.default_rng(cfg.seed)
    n_envs = model.replay_buffer.n_envs

    env = RobosuiteLiftEnv(cfg.env)
    successes_added = 0
    attempt = 0
    try:
        env.reset()
        if "cube_pos" not in env.last_obs_dict or "robot0_eef_pos" not in env.last_obs_dict:
            LOGGER.warning(
                "Sparse-reward replay seeding: obs dict has no 'cube_pos'/"
                "'robot0_eef_pos' (available: %s). Scripted heuristic can't run for "
                "this task/robot; skipping seeding.",
                sorted(env.last_obs_dict),
            )
            return 0

        for attempt in range(max_attempts):
            if successes_added >= target_successes:
                break
            transitions, succeeded = _rollout_scripted_episode(env, rng)
            if not succeeded:
                continue
            for obs, next_obs, action, reward, terminated, truncated in transitions:
                model.replay_buffer.add(
                    np.tile(obs[None], (n_envs, 1)),
                    np.tile(next_obs[None], (n_envs, 1)),
                    np.tile(action[None], (n_envs, 1)),
                    np.full(n_envs, reward, dtype=np.float32),
                    np.full(n_envs, terminated or truncated),
                    [{"TimeLimit.truncated": truncated and not terminated}] * n_envs,
                )
            successes_added += 1
    finally:
        env.close()

    if successes_added == 0:
        LOGGER.warning(
            "Sparse-reward replay seeding found 0 successes in %d scripted "
            "attempts; the replay buffer starts empty as usual.",
            attempt + 1,
        )
    else:
        LOGGER.info(
            "Sparse-reward replay seeding: added %d successful scripted episode(s) "
            "to the replay buffer (%d attempts).",
            successes_added,
            attempt + 1,
        )
    return successes_added
