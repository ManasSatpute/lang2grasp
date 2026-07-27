#!/usr/bin/env python3
"""Unit-level checks for the Phase 2 paradigm-switch building blocks (see
scripts/train_paradigm.py's module docstring): the continuous ObjectParams sampler,
the z-vector encoding pi_param FiLM-conditions on, and the GRU/FiLM feature
extractors pi_blind+hist/pi_param use.

Deliberately does *not* touch robosuite or a real env: `objects.object_params` has no
simulator dependency, and `rl.policies`'s extractors are plain torch modules that only
need the right tensor shapes -- so this runs anywhere torch/stable-baselines3 are
installed, without a MuJoCo/robosuite stack (unlike smoke_test.py/force_sensor_test.py).

Usage (from the repo root):
    PYTHONPATH=src python src/tests/paradigm_test.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from objects.object_params import (  # noqa: E402
    Z_DIM,
    ObjectParams,
    _SIZE_BOUNDS_M,  # private, but the test needs the exact sampling bounds to check against
    object_params_to_noisy_z,
    object_params_to_z,
    sample_object_params,
)
from rl.policies import FiLMExtractor, HistoryGRUExtractor  # noqa: E402
from common.utils import setup_logging  # noqa: E402

LOGGER = logging.getLogger("paradigm_test")


class _FakeSpace:
    """Stand-in for a gymnasium Box: the extractors only ever read `.shape`."""

    def __init__(self, dim: int) -> None:
        self.shape = (dim,)


def test_sample_object_params_within_bounds() -> None:
    """[1] Every sampled object is already inside the clamp ranges __post_init__
    enforces -- i.e. sample_object_params never produces something that needed
    clamping (a silent sign its ranges have drifted from _SIZE_BOUNDS_M et al.)."""
    seen_shapes = set()
    for _ in range(200):
        params = sample_object_params()
        seen_shapes.add(params.shape)
        bounds = _SIZE_BOUNDS_M[params.shape]
        for value, (lo, hi) in zip(params.size, bounds):
            assert lo <= value <= hi, f"{params.shape} size {value} outside [{lo},{hi}]"
    assert seen_shapes == {"box", "cylinder", "ball"}, f"shapes never sampled: {seen_shapes}"

    only_boxes = [sample_object_params(shapes=("box",)) for _ in range(20)]
    assert all(p.shape == "box" for p in only_boxes), "shapes= restriction not respected"
    LOGGER.info("[1] OK: sample_object_params stays in-bounds and honours shapes=.")


def test_z_vector_excludes_mass_but_recovers_it() -> None:
    """[2] z has exactly Z_DIM entries, and mass_kg -- deliberately excluded from z
    (it's density * volume_m3, collinear with dims already in z) -- is still exactly
    recoverable from z's own density + size, proving nothing was lost by dropping it."""
    for shape, size in (
        ("box", (0.02, 0.03, 0.015)),
        ("cylinder", (0.02, 0.05)),
        ("ball", (0.025,)),
    ):
        params = ObjectParams(name="t", shape=shape, size=size, density=1200.0)
        z = object_params_to_z(params)
        assert z.shape == (Z_DIM,), f"{shape}: expected z.shape=({Z_DIM},), got {z.shape}"
        assert z.dtype == np.float32

        n_shapes = 3
        one_hot = z[:n_shapes]
        assert one_hot.sum() == 1.0 and one_hot[["box", "cylinder", "ball"].index(shape)] == 1.0

        size_padded = z[n_shapes : n_shapes + 3]
        assert np.allclose(size_padded[: len(size)], size)
        assert np.all(size_padded[len(size) :] == 0.0), "unused size slots must be zero-padded"

        density_from_z = float(z[n_shapes + 3])
        assert density_from_z == np.float32(params.density)
        # Recompute volume from z's own (shape, padded size) -- same formula
        # ObjectParams.volume_m3 uses -- and check it reconstructs mass_kg exactly.
        if shape == "box":
            volume = 8.0 * size_padded[0] * size_padded[1] * size_padded[2]
        elif shape == "cylinder":
            volume = np.pi * size_padded[0] ** 2 * (2.0 * size_padded[1])
        else:
            volume = (4.0 / 3.0) * np.pi * size_padded[0] ** 3
        recovered_mass = density_from_z * float(volume)
        assert abs(recovered_mass - params.mass_kg) / params.mass_kg < 1e-3, (
            f"{shape}: mass recoverable from z ({recovered_mass}) != params.mass_kg ({params.mass_kg})"
        )
    LOGGER.info("[2] OK: z is Z_DIM-wide, mass-free, and mass is exactly recoverable from it.")


def test_noisy_z_perturbs_only_continuous_dims() -> None:
    """[3] rel_noise_std=0.0 is the exact z; rel_noise_std>0.0 leaves the one-hot shape
    block exact but (almost certainly, over many draws) perturbs every continuous dim."""
    params = ObjectParams(name="t", shape="box", size=(0.02, 0.02, 0.02), density=1000.0)
    exact = object_params_to_z(params)

    noiseless = object_params_to_noisy_z(params, rel_noise_std=0.0)
    assert np.array_equal(noiseless, exact), "rel_noise_std=0.0 must return the exact z"

    any_perturbed = np.zeros(Z_DIM - 3, dtype=bool)
    for _ in range(50):
        noisy = object_params_to_noisy_z(params, rel_noise_std=0.2)
        assert np.array_equal(noisy[:3], exact[:3]), "one-hot shape dims must never be perturbed"
        any_perturbed |= np.asarray(noisy[3:] != exact[3:])
    assert any_perturbed.all(), "every continuous dim should get perturbed over 50 draws"
    LOGGER.info("[3] OK: noisy z leaves shape exact and perturbs every continuous dim.")


def test_history_gru_extractor_shapes() -> None:
    """[4] HistoryGRUExtractor splits (current, history) correctly and outputs
    features_dim = current_dim + gru_hidden; a mismatched obs dim raises."""
    current_dim, history_len, step_dim, gru_hidden, batch = 20, 16, 9, 32, 4
    obs_dim = current_dim + history_len * step_dim
    extractor = HistoryGRUExtractor(
        _FakeSpace(obs_dim),
        current_dim=current_dim,
        history_len=history_len,
        step_dim=step_dim,
        gru_hidden=gru_hidden,
    )
    assert extractor.features_dim == current_dim + gru_hidden

    obs = th.randn(batch, obs_dim)
    out = extractor(obs)
    assert out.shape == (batch, current_dim + gru_hidden), out.shape
    assert th.equal(out[:, :current_dim], obs[:, :current_dim]), "current-obs slice must pass through unchanged"

    try:
        HistoryGRUExtractor(
            _FakeSpace(obs_dim + 1), current_dim=current_dim, history_len=history_len, step_dim=step_dim
        )
        raise AssertionError("expected ValueError on a mismatched observation dim")
    except ValueError:
        pass
    LOGGER.info("[4] OK: HistoryGRUExtractor shapes/pass-through/validation all correct.")


def test_film_extractor_conditions_on_z() -> None:
    """[5] FiLMExtractor outputs features_dim=hidden_dim, and z actually modulates
    the output (not silently ignored) -- same state, different z -> different output."""
    z_dim, hidden_dim, state_dim, batch = Z_DIM, 64, 30, 4
    obs_dim = state_dim + z_dim
    extractor = FiLMExtractor(_FakeSpace(obs_dim), z_dim=z_dim, hidden_dim=hidden_dim)
    assert extractor.features_dim == hidden_dim

    state = th.randn(batch, state_dim)
    z_a = th.randn(batch, z_dim)
    z_b = th.randn(batch, z_dim)
    out_a = extractor(th.cat([state, z_a], dim=1))
    out_b = extractor(th.cat([state, z_b], dim=1))
    assert out_a.shape == (batch, hidden_dim)
    assert not th.allclose(out_a, out_b), "different z must change the output (FiLM not a no-op)"

    try:
        FiLMExtractor(_FakeSpace(z_dim), z_dim=z_dim, hidden_dim=hidden_dim)
        raise AssertionError("expected ValueError when obs_dim <= z_dim")
    except ValueError:
        pass
    LOGGER.info("[5] OK: FiLMExtractor shapes correct and z demonstrably modulates output.")


def main() -> int:
    setup_logging()
    test_sample_object_params_within_bounds()
    test_z_vector_excludes_mass_but_recovers_it()
    test_noisy_z_perturbs_only_continuous_dims()
    test_history_gru_extractor_shapes()
    test_film_extractor_conditions_on_z()
    LOGGER.info("PARADIGM TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
