"""Replay-buffer seeding for the sparse/"fixed" reward config (``EnvConfig.reward_shaping=False``).

With ``reward_shaping=False``, robosuite's ``Lift`` gives a fixed reward: 0.0 every
step until the cube crosses the success height, then a constant positive value (see
``EnvConfig.reward_shaping``'s docstring in ``rl/env.py`` and README.md's "The sparse
default reward" section). A freshly-initialised SAC policy essentially never lifts a
cube by chance with a 7-DoF arm, so the replay buffer fills with all-zero-reward
transitions and the critic never sees a gradient -- hence the ~0% success rate.

This module runs a scripted, non-learned reach -> descend -> grasp -> lift heuristic
against a throwaway single env to *find* a handful of real successes before training
starts, and inserts their transitions directly into SAC's replay buffer so the critic
has real reward signal to bootstrap from. It only ever runs when the caller
(``rl/train.py``) checks ``cfg.env.reward_shaping is False`` first -- the shaped
(default) path never imports or calls this.
"""

from __future__ import annotations

import logging

import numpy as np
from stable_baselines3 import SAC

from rl.config import TrainConfig
from rl.env import RobosuiteLiftEnv

LOGGER = logging.getLogger(__name__)

_HOVER_HEIGHT_M = 0.08
_XY_TOL_M = 0.015
_DESCEND_TOL_M = 0.012
_GRASP_HOLD_STEPS = 10
_GAIN = 4.0
_MIN_ACTION_DIM = 4  # 3 position deltas + >=1 gripper dim


class _ScriptedPickPolicy:
    """Proportional-control reach/descend/grasp/lift heuristic. Not a learned policy.

    Just needs to succeed occasionally so the replay buffer gets non-zero-reward
    transitions. Reads world-frame ``cube_pos`` / ``robot0_eef_pos`` off the raw
    robosuite obs dict (``RobosuiteLiftEnv.last_obs_dict``); degrades gracefully
    (``act`` returns ``None``) if either key is missing, e.g. a task/robot this
    wasn't written against.
    """

    def __init__(
        self, action_low: np.ndarray, action_high: np.ndarray, rng: np.random.Generator
    ) -> None:
        self._low = action_low
        self._high = action_high
        self._hover_height = _HOVER_HEIGHT_M * float(rng.uniform(0.7, 1.3))
        self._gain = _GAIN * float(rng.uniform(0.8, 1.2))
        self._phase = "approach"
        self._grasp_steps = 0

    def act(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray | None:
        cube_pos = obs_dict.get("cube_pos")
        eef_pos = obs_dict.get("robot0_eef_pos")
        if cube_pos is None or eef_pos is None:
            return None
        cube_pos = np.asarray(cube_pos, dtype=np.float64)
        eef_pos = np.asarray(eef_pos, dtype=np.float64)
        xy_err = float(np.linalg.norm(cube_pos[:2] - eef_pos[:2]))

        if self._phase == "approach":
            target = cube_pos + np.array([0.0, 0.0, self._hover_height])
            if xy_err < _XY_TOL_M:
                self._phase = "descend"
            gripper = -1.0
        elif self._phase == "descend":
            target = cube_pos
            if xy_err < _DESCEND_TOL_M and abs(cube_pos[2] - eef_pos[2]) < _DESCEND_TOL_M:
                self._phase = "grasp"
            gripper = -1.0
        elif self._phase == "grasp":
            target = eef_pos  # hold position while the fingers close
            self._grasp_steps += 1
            if self._grasp_steps >= _GRASP_HOLD_STEPS:
                self._phase = "lift"
            gripper = 1.0
        else:  # lift
            target = eef_pos + np.array([0.0, 0.0, 0.05])
            gripper = 1.0

        action = np.zeros_like(self._low)
        action[:3] = self._gain * (target - eef_pos)
        action[-1] = gripper
        return np.clip(action, self._low, self._high)


def _rollout_scripted_episode(
    env: RobosuiteLiftEnv, rng: np.random.Generator
) -> tuple[list[tuple], bool]:
    """One scripted episode. Returns (transitions, succeeded)."""
    low, high = env.action_space.low, env.action_space.high
    if low.shape[0] < _MIN_ACTION_DIM:
        return [], False

    policy = _ScriptedPickPolicy(low, high, rng)
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
