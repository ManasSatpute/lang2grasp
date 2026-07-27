"""Custom SB3 feature extractors for the paradigm-switch policy variants.

Three variants are trained against the same domain-randomized env
(``EnvConfig.randomize_object``, see ``scripts/train_paradigm.py``):

- pi_blind: stock ``MlpPolicy`` (SB3's default ``FlattenExtractor``) -- no custom
  extractor needed.
- pi_blind+hist: :class:`HistoryGRUExtractor`, paired with ``rl.env.HistoryObsWrapper``
  (``EnvConfig.history_len > 0``). Memoryless pi_blind can't tell objects apart within
  an episode; this variant can do implicit system identification from how the arm's
  proprioception/force responded over the last few steps.
- pi_param: :class:`FiLMExtractor`, paired with ``EnvConfig.include_object_z``. Told
  (approximately -- see ``objects.object_params.object_params_to_noisy_z``) what
  object it's dealing with, rather than having to infer it.

Per-object specialists (``scripts/train_object.py``/``train_all_objects.py``) use none
of these -- they stay stock ``MlpPolicy``, since they're the oracle topline, not a
paradigm variant.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from objects.object_params import Z_DIM


class HistoryGRUExtractor(BaseFeaturesExtractor):
    """pi_blind+hist's feature extractor.

    Input layout (see ``rl.env.HistoryObsWrapper``): ``[current obs (current_dim,),
    history block (history_len * step_dim,)]``, history oldest-first. Splits the two
    apart, runs a GRU over the history block, and returns
    ``[current obs, GRU final hidden state]`` -- SB3's own ``net_arch`` MLP (SAC's
    actor/critic heads) does everything downstream of that, exactly as it would on
    top of a plain ``MlpPolicy``.

    ``current_dim``/``history_len``/``step_dim`` must match the
    ``HistoryObsWrapper``/``EnvConfig`` the env was actually built with -- see
    ``rl.env.history_dims``, which derives them from an ``EnvConfig`` instead of
    requiring them to be hand-computed and kept in sync.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        current_dim: int,
        history_len: int,
        step_dim: int,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim=current_dim + gru_hidden)
        expected_dim = current_dim + history_len * step_dim
        actual_dim = observation_space.shape[0]
        if actual_dim != expected_dim:
            raise ValueError(
                f"HistoryGRUExtractor expected obs dim {expected_dim} "
                f"(current_dim={current_dim} + history_len={history_len} * step_dim={step_dim}), "
                f"got {actual_dim}. These must match the HistoryObsWrapper/EnvConfig the env "
                "was actually built with -- see rl.env.history_dims."
            )
        self.current_dim = current_dim
        self.history_len = history_len
        self.step_dim = step_dim
        self.gru = nn.GRU(input_size=step_dim, hidden_size=gru_hidden, batch_first=True)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        current = observations[:, : self.current_dim]
        history = observations[:, self.current_dim :].reshape(-1, self.history_len, self.step_dim)
        _, h_n = self.gru(history)
        gru_out = h_n[-1]  # (batch, gru_hidden) -- last layer's final hidden state
        return th.cat([current, gru_out], dim=1)


class FiLMExtractor(BaseFeaturesExtractor):
    """pi_param's feature extractor: FiLM-conditions an MLP trunk over the ordinary
    state observation on a (possibly noisy) object-parameter vector z.

    Input layout (see ``EnvConfig.include_object_z``/``rl.env.OBJECT_Z_KEY``): z is
    always the *last* ``z_dim`` entries of the observation, since
    ``RobosuiteLiftEnv.__init__`` appends ``OBJECT_Z_KEY`` after every other
    configured ``obs_key``. ``gamma``/``beta`` modulate the trunk's hidden
    activations -- ``h' = gamma * h + beta`` (Perez et al., 2018, FiLM).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        z_dim: int = Z_DIM,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=hidden_dim)
        obs_dim = observation_space.shape[0]
        if obs_dim <= z_dim:
            raise ValueError(
                f"FiLMExtractor: observation dim ({obs_dim}) must exceed z_dim ({z_dim}) -- "
                "is EnvConfig.include_object_z actually set on this env?"
            )
        self.z_dim = z_dim
        self.state_dim = obs_dim - z_dim
        self.trunk = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.ReLU())
        self.film = nn.Linear(z_dim, 2 * hidden_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        state = observations[:, : self.state_dim]
        z = observations[:, self.state_dim :]
        h = self.trunk(state)
        gamma, beta = self.film(z).chunk(2, dim=1)
        return gamma * h + beta
