"""Typed configuration objects, loaded from JSON with CLI overrides."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rl.env import EnvConfig

Algo = Literal["PPO", "SAC"]


@dataclass
class TrainConfig:
    """Hyper-parameters and run plumbing. Everything here lands in the run snapshot."""

    algo: Algo = "SAC"
    seed: int = 0
    total_timesteps: int = 1_000_000
    n_envs: int = 1
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    #: Threads for torch in the learner process. Keep low: each SubprocVecEnv worker
    #: also links BLAS, and n_envs x OMP threads will thrash a 12-core slot.
    n_threads: int = 2

    #: VecNormalize + an off-policy replay buffer is a footgun: stored transitions
    #: were normalised with statistics that keep drifting. On for PPO, off for SAC.
    normalize_obs: bool = False
    normalize_reward: bool = False

    checkpoint_freq: int = 50_000  # in env steps; divided by n_envs internally
    eval_freq: int = 25_000
    n_eval_episodes: int = 10

    #: Persist the SAC replay buffer so a requeued job resumes without a cold restart.
    #: Written once on exit, not every checkpoint: the pickle is hundreds of MB.
    save_replay_buffer: bool = True

    #: Stop and checkpoint after this many hours, before SLURM's hard kill. Leave
    #: ~15 min of headroom below the `#SBATCH --time` value. null = no limit.
    max_hours: float | None = None
    #: Resume from the newest checkpoint in the run dir if one exists.
    resume: bool = False

    log_dir: str = "runs"
    run_name: str | None = None

    #: Passed verbatim to the SB3 constructor. Keys must match the algo's signature.
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    algo_kwargs: dict[str, Any] = field(default_factory=dict)

    env: EnvConfig = field(default_factory=EnvConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate a dataclass, rejecting unknown keys instead of silently dropping them."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a JSON config file and apply flat top-level overrides from the CLI."""
    raw: dict[str, Any] = json.loads(Path(path).read_text()) or {}
    env_payload = raw.pop("env", {}) or {}

    for key, value in (overrides or {}).items():
        if value is not None:
            raw[key] = value

    cfg = _build(TrainConfig, raw)
    cfg.env = _build(EnvConfig, env_payload)
    return cfg


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    """Snapshot the resolved config next to the checkpoints, for reproducibility.

    Never overwrite an existing snapshot: a requeued job must keep the config it
    was originally launched with, not whatever the JSON says today.
    """
    target = Path(path)
    if target.exists():
        return
    target.write_text(json.dumps(cfg.to_dict(), indent=2))
