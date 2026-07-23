"""Load a saved policy and roll it out -- the round-trip sanity check.

A checkpoint is only "saved" if it can be reloaded into a fresh process and
reproduce the behaviour it had at save time. This script verifies:

1. The archive deserialises and its spaces match a freshly built env.
2. The policy actually acts: episodes complete, success rate is reported.

Usage (from the repo root):
    PYTHONPATH=src python -m rl.rollout --run-dir runs/lift_123 --episodes 10
    PYTHONPATH=src python -m rl.rollout --run-dir runs/lift_123 \
        --model eval/best_model.zip --video out.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from rl.config import TrainConfig, load_config
from rl.env import EnvConfig
from objects.object_params import ObjectParams
from common.utils import (
    CONFIG_SNAPSHOT,
    FINAL_MODEL_NAME,
    VECNORM_NAME,
    resolve_device,
    setup_logging,
)
from rl.vec_env import build_vec_env

LOGGER = logging.getLogger(__name__)


def load_policy(
    run_dir: Path, model_rel: str, device: str = "auto"
) -> tuple[BaseAlgorithm, TrainConfig]:
    """Reload ``cfg`` from the run snapshot, then the matching SB3 archive."""
    cfg = load_config(run_dir / CONFIG_SNAPSHOT)
    model_path = run_dir / model_rel
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint at {model_path}")

    model = SAC.load(model_path, device=resolve_device(device))
    LOGGER.info("Loaded SAC from %s", model_path)
    return model, cfg


def load_run_object_params(run_dir: Path) -> ObjectParams | None:
    """The `ObjectParams` a run's `config.json` snapshot carries, or None for the
    stock-cube baseline (`env.object` unset at train time)."""
    config = json.loads((run_dir / CONFIG_SNAPSHOT).read_text())
    object_payload = config.get("env", {}).get("object")
    return ObjectParams(**object_payload) if object_payload else None


def rollout(
    run_dir: Path,
    model_rel: str = FINAL_MODEL_NAME,
    episodes: int = 10,
    seed: int = 1234,
    device: str = "auto",
    video_path: Path | None = None,
    object_override: ObjectParams | None = None,
    model: BaseAlgorithm | None = None,
    cfg: TrainConfig | None = None,
) -> dict[str, float]:
    """Roll the loaded policy out and return aggregate metrics.

    ``model``/``cfg`` let a caller pass an already-loaded policy (via
    :func:`load_policy`) instead of reloading it from ``run_dir`` -- useful when
    rolling the same model out repeatedly, e.g. against several ``object_override``s.

    ``object_override`` evaluates the policy against a *different* object's physics
    than the one ``run_dir`` was trained on: the env is rebuilt from the run's own
    ``cfg.env`` (matching horizon/controller/obs_keys/etc.) with only ``object``
    replaced. Comparing a generically-trained policy (``run_dir``'s own ``env.object``
    is ``None``) against each LLM-described object's real physics is the main use
    case -- see `scripts/compare_policies.py`. If the run used
    ``normalize_obs``/``normalize_reward``, its `VecNormalize` stats were fit on the
    *original* object's dynamics, so they're a mismatch for the override; harmless for
    the default config (`normalize_obs: false`), but worth knowing.
    """
    if model is None or cfg is None:
        model, cfg = load_policy(run_dir, model_rel, device)

    env_cfg: EnvConfig = cfg.env
    if video_path is not None or object_override is not None:
        overrides: dict[str, object] = {}
        if video_path is not None:
            overrides["render_mode"] = "rgb_array"
        if object_override is not None:
            overrides["object"] = object_override
        env_cfg = EnvConfig(**{**cfg.env.__dict__, **overrides})

    vecnorm = run_dir / VECNORM_NAME
    env = build_vec_env(
        cfg,
        n_envs=1,
        seed=seed,
        training=False,  # freeze the running mean/std; do not normalise reward
        env_cfg=env_cfg,
        vecnormalize_path=vecnorm if vecnorm.exists() else None,
    )

    # The reloaded policy and a fresh env must agree on their spaces.
    assert model.observation_space.shape == env.observation_space.shape, (
        f"obs space mismatch: {model.observation_space} vs {env.observation_space}"
    )
    assert model.action_space.shape == env.action_space.shape, "action space mismatch"

    returns: list[float] = []
    lengths: list[int] = []
    held: list[bool] = []  # cube lifted at the FINAL step -- matches eval/success_rate
    ever: list[bool] = []  # cube lifted at ANY step -- a much weaker claim
    frames: list[np.ndarray] = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_return, ep_len = 0.0, 0
        ep_ever, step_success = False, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            ep_return += float(reward[0])
            ep_len += 1
            # SB3's VecEnv preserves the terminal step's info on the `done` step,
            # so the last value we see is the terminal one.
            step_success = bool(infos[0].get("is_success", False))
            ep_ever |= step_success
            done = bool(dones[0])
            if video_path is not None:
                frames.append(env.env_method("render", indices=0)[0])

        returns.append(ep_return)
        lengths.append(ep_len)
        held.append(step_success)  # value at the terminal step
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

    env.close()

    if video_path is not None and frames:
        import imageio.v2 as imageio

        imageio.mimwrite(video_path, frames, fps=cfg.env.control_freq)
        LOGGER.info("Wrote %d frames to %s", len(frames), video_path)

    n = len(returns)
    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        # THE headline number. Cube held at the terminal step -- the same definition
        # SB3's EvalCallback uses, so it is directly comparable to eval/success_rate.
        "success_rate": float(np.mean(held)),
        # Cube lifted at any point. Always >= success_rate. A policy that grabs and
        # drops scores 1.0 here and 0.0 above, so never quote this one alone.
        "success_ever": float(np.mean(ever)),
        # Binomial standard error. With 20 episodes a rate of 0.30 carries ~+/-0.10,
        # so a 30% -> 20% swing between evals is noise, not regression.
        "success_stderr": float(np.sqrt(np.mean(held) * (1 - np.mean(held)) / max(1, n))),
        "n_episodes": n,
    }
    LOGGER.info("Rollout summary: %s", metrics)
    if metrics["success_ever"] > metrics["success_rate"] + 1e-9:
        LOGGER.warning(
            "Cube was lifted in %.0f%% of episodes but still held at the end in only "
            "%.0f%%. The policy grasps and drops.",
            100 * metrics["success_ever"],
            100 * metrics["success_rate"],
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--model", dest="model_rel", default=FINAL_MODEL_NAME)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video", dest="video_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    rollout(**vars(parse_args()))


if __name__ == "__main__":
    main()
