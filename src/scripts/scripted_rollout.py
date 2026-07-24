"""Roll out the scripted (non-learned) pick heuristic directly -- no SAC, no training,
no ``ObjectParams``. Useful to check whether a plain reach/descend/grasp/lift controller
can lift the cube on its own, independent of whatever the RL policy is or isn't
learning under the sparse/"fixed" reward (see ``rl/sparse_seed.py``, README.md's "The
sparse default reward" section).

Metrics reported match ``rl.rollout.rollout``'s: ``success_rate`` is the cube held at
the terminal step, ``success_ever`` is the cube lifted at any point (always >= the
former; a policy that grasps and drops scores 1.0 there, 0.0 here).

Usage (from the repo root):
    PYTHONPATH=src python src/scripts/scripted_rollout.py --episodes 20
    PYTHONPATH=src python src/scripts/scripted_rollout.py --episodes 5 --render
    PYTHONPATH=src python src/scripts/scripted_rollout.py --episodes 5 --video out.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from common.utils import setup_logging
from rl.env import EnvConfig, RobosuiteLiftEnv
from rl.scripted_policy import MIN_ACTION_DIM, ScriptedPickPolicy

LOGGER = logging.getLogger(__name__)


def rollout_scripted(
    episodes: int = 10,
    seed: int = 1234,
    horizon: int = 500,
    render: bool = False,
    video_path: Path | None = None,
) -> dict[str, float]:
    """Run the scripted heuristic for ``episodes`` episodes and return aggregate metrics."""
    if render and video_path is not None:
        raise ValueError(
            "render and video_path are mutually exclusive -- render_mode is either "
            "'human' (live window) or 'rgb_array' (offscreen, for video), not both."
        )

    render_mode = "human" if render else ("rgb_array" if video_path is not None else None)
    env_cfg = EnvConfig(horizon=horizon, render_mode=render_mode)
    env = RobosuiteLiftEnv(env_cfg)
    rng = np.random.default_rng(seed)

    returns: list[float] = []
    lengths: list[int] = []
    held: list[bool] = []
    ever: list[bool] = []
    frames: list[np.ndarray] = []

    try:
        low, high = env.action_space.low, env.action_space.high
        if low.shape[0] < MIN_ACTION_DIM:
            raise SystemExit(
                f"Action space has only {low.shape[0]} dim(s); the scripted heuristic "
                f"needs at least {MIN_ACTION_DIM} (3 position deltas + a gripper)."
            )

        for ep in range(episodes):
            obs, _ = env.reset()
            if "cube_pos" not in env.last_obs_dict or "robot0_eef_pos" not in env.last_obs_dict:
                raise SystemExit(
                    "obs dict has no 'cube_pos'/'robot0_eef_pos' (available: "
                    f"{sorted(env.last_obs_dict)}). The scripted heuristic can't run "
                    "for this task/robot."
                )
            policy = ScriptedPickPolicy(low, high, rng)
            terminated = truncated = False
            ep_return, ep_len = 0.0, 0
            ep_ever, step_success = False, False

            while not (terminated or truncated):
                action = policy.act(env.last_obs_dict)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                step_success = bool(info.get("is_success", False))
                ep_ever |= step_success
                if video_path is not None:
                    frames.append(env.render())
                elif render:
                    env.render()

            returns.append(ep_return)
            lengths.append(ep_len)
            held.append(step_success)
            ever.append(ep_ever)
            LOGGER.info(
                "episode %2d/%d | return %8.2f | len %3d | held %-5s | ever %-5s",
                ep + 1,
                episodes,
                ep_return,
                ep_len,
                step_success,
                ep_ever,
            )
    finally:
        env.close()

    if video_path is not None and frames:
        import imageio.v2 as imageio

        imageio.mimwrite(video_path, frames, fps=env_cfg.control_freq)
        LOGGER.info("Wrote %d frames to %s", len(frames), video_path)

    n = len(returns)
    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(held)),
        "success_ever": float(np.mean(ever)),
        "success_stderr": float(np.sqrt(np.mean(held) * (1 - np.mean(held)) / max(1, n))),
        "n_episodes": n,
    }
    LOGGER.info("Scripted rollout summary: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--video", dest="video_path", type=Path, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Live MuJoCo viewer window. Needs a display; mutually exclusive with --video.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    rollout_scripted(**vars(parse_args()))


if __name__ == "__main__":
    main()
