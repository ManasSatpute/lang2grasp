#!/usr/bin/env python3
"""End-to-end check for the genuine fingertip force sensor (see rl/env.py's module
docstring and objects/force_gripper.py). Proves four things:

  [1] `fingertip_force` is part of the observation for the generic baseline too.
  [2] The reading is a genuine sensor value that changes once a scripted grasp
      actually makes contact, not a constant.
  [3] Crushing is real: a low crush_force_N terminates the episode with a reward
      penalty.
  [4] The width/mass-matched object generator reproduces its target width_mm/mass_g.
  [5] Crushing is *only* the gripper on the object: force from anything else (a pad
      pressed into the tabletop) never terminates the episode, however low
      crush_force_N is.

Needs a real robosuite/mujoco install, but not stable-baselines3/torch.

Usage (from the repo root):
    PYTHONPATH=src python src/tests/force_sensor_test.py
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.env import RobosuiteLiftEnv, EnvConfig  # noqa: E402
from rl.scripted_policy import ScriptedPickPolicy  # noqa: E402
from objects.object_params import ObjectParams  # noqa: E402
from scripts.generate_width_mass_objects import generate_objects  # noqa: E402
from common.utils import setup_logging  # noqa: E402

LOGGER = logging.getLogger("force_sensor_test")

_WIDTH_MASS_OBJECT = Path("src/configs/objects/width_mass_set/width40_mass200g.json")


def _run_scripted_episode(env: RobosuiteLiftEnv, max_steps: int, seed: int = 0):
    """Roll a scripted grasp out; returns (fingertip_force history, final info)."""
    policy = ScriptedPickPolicy(env.action_space.low, env.action_space.high, np.random.default_rng(seed))
    env.reset()
    forces = []
    info: dict = {}
    for _ in range(max_steps):
        action = policy.act(env.last_obs_dict)
        if action is None:
            break
        _, _, terminated, truncated, info = env.step(action)
        forces.append(env.last_obs_dict["fingertip_force"].copy())
        if terminated or truncated:
            break
    return forces, info


def assert_baseline_obs_includes_force() -> None:
    """[1] The generic baseline (no ObjectParams) still carries fingertip_force."""
    env = RobosuiteLiftEnv(EnvConfig(horizon=20))
    try:
        obs, _ = env.reset()
        assert "fingertip_force" in env.last_obs_dict, "baseline obs_dict missing 'fingertip_force'"
        force = env.last_obs_dict["fingertip_force"]
        assert force.shape == (6,), f"expected shape (6,), got {force.shape}"

        non_force_dim = obs.shape[0] - force.size
        env_no_force = RobosuiteLiftEnv(EnvConfig(horizon=20, obs_keys=("robot0_proprio-state", "object-state")))
        try:
            assert non_force_dim == env_no_force.observation_space.shape[0], (
                f"obs_dim without 'fingertip_force' should match the sensor-free obs_keys "
                f"({non_force_dim} vs {env_no_force.observation_space.shape[0]})"
            )
        finally:
            env_no_force.close()
    finally:
        env.close()
    LOGGER.info("[1] OK: baseline observation includes a (6,) fingertip_force reading.")


def assert_force_reading_is_genuine() -> None:
    """[2] The reading changes once the scripted grasp makes contact -- not a constant."""
    params = ObjectParams(**json.loads(_WIDTH_MASS_OBJECT.read_text()))
    env = RobosuiteLiftEnv(EnvConfig(horizon=300, object=params))
    try:
        forces, info = _run_scripted_episode(env, max_steps=300)
        assert forces, "scripted policy produced no steps -- can't check the sensor"
        magnitudes = [float(np.linalg.norm(f)) for f in forces]
        assert max(magnitudes) > min(magnitudes) + 1e-6, (
            "fingertip_force never changed across the episode -- looks like a "
            "constant/stub reading, not a real sensor"
        )
        # A real lift check: ParamLift._check_success measures height against the
        # object's *own* resting height, so this no longer passes for free on an object
        # (like this 45mm-half-height cylinder) that is taller than robosuite's stock cube.
        assert info.get("is_success"), f"scripted grasp did not succeed against {params.name}: {info}"
    finally:
        env.close()
    LOGGER.info(
        "[2] OK: fingertip_force varies during the episode (range %.4f-%.4fN) and the "
        "grasp succeeded.",
        min(magnitudes),
        max(magnitudes),
    )


def assert_crush_terminates() -> None:
    """[3] A near-zero crush_force_N causes real termination with a reward penalty."""
    base = ObjectParams(**json.loads(_WIDTH_MASS_OBJECT.read_text()))
    fragile = dataclasses.replace(base, crush_force_N=0.1)  # ObjectParams' own clamp floor
    env = RobosuiteLiftEnv(EnvConfig(horizon=300, object=fragile))
    try:
        policy = ScriptedPickPolicy(env.action_space.low, env.action_space.high, np.random.default_rng(0))
        env.reset()
        crushed = False
        crush_force_N = crush_reward = None
        uncrushed_reward = None  # the plain robosuite shaped reward, for the "penalty applied" comparison below
        for _ in range(300):
            action = policy.act(env.last_obs_dict)
            if action is None:
                break
            _, reward, terminated, truncated, info = env.step(action)
            if info.get("crushed"):
                crushed = True
                crush_force_N = info["fingertip_force_N"]
                crush_reward = reward
                assert terminated, "info['crushed'] was True but terminated was False"
                break
            uncrushed_reward = reward
            if terminated or truncated:
                break
        assert crushed, (
            "expected crush_force_N=0.1 to be exceeded and terminate the episode, but "
            "it never did -- crush penalty/termination logic may be broken"
        )
        assert crush_force_N > fragile.crush_force_N, (
            f"episode was flagged crushed but the recorded fingertip_force_N "
            f"({crush_force_N}) doesn't actually exceed crush_force_N "
            f"({fragile.crush_force_N})"
        )
        # The crush penalty (crush_penalty_coeff * excess, e.g. 0.1 * ~0.05N here) is
        # deliberately tiny relative to the shaped reward's usual +0.5..+2 scale for a
        # threshold this low -- crush_force_N=0.1 is an artificially small worst-case
        # to make crushing trivially reachable, not a realistic object. So this checks
        # the penalty was *applied* (reward measurably lower than an uncrushed step),
        # not that it flipped the overall sign.
        if uncrushed_reward is not None:
            assert crush_reward < uncrushed_reward, (
                f"crushed-step reward ({crush_reward}) should be lower than a prior "
                f"uncrushed step's ({uncrushed_reward}) -- penalty doesn't look applied"
            )
    finally:
        env.close()
    LOGGER.info(
        "[3] OK: crush_force_N=0.1 terminated the episode (force=%.4fN) with reward %.4f.",
        crush_force_N,
        crush_reward,
    )


def assert_crush_ignores_non_object_contact() -> None:
    """[5] Force that isn't the gripper squeezing the object -- here, the closed pads
    pressed into the tabletop away from it -- must never register as a crush, however
    low crush_force_N is.

    This is the regression that pinned `eval/success_rate` at 0 for every per-object
    run: the crush check read the raw fingertip sensor with no contact gate, so with a
    `crush_force_N` of 5-12N (every fragile object, and all of width_mass_set at 10N) a
    table bump during the reach phase terminated the episode after a handful of steps
    and training never got near a grasp.
    """
    base = ObjectParams(**json.loads(_WIDTH_MASS_OBJECT.read_text()))
    fragile = dataclasses.replace(base, crush_force_N=0.1)  # ObjectParams' own clamp floor
    env = RobosuiteLiftEnv(EnvConfig(horizon=200, object=fragile))
    try:
        env.reset()
        action_dim = env.action_space.shape[0]
        max_off_object_force_N = 0.0
        for step in range(120):
            action = np.zeros(action_dim, dtype=np.float64)
            if step < 30:
                action[1] = 0.5  # translate clear of the cube's placement region
            else:
                action[2] = -1.0  # then drive the closed pads down into the tabletop
            action[-1] = 1.0  # gripper closed throughout
            _, _, terminated, truncated, info = env.step(action)

            # Same post-step sim state env.step() itself read, so this is exactly the
            # contact the crush check saw.
            if any(env._pads_touching_object()):
                continue  # a genuine gripper/object contact -- not what this test covers
            raw_N = info["fingertip_force_raw_N"]
            max_off_object_force_N = max(max_off_object_force_N, raw_N)
            assert not info["crushed"], (
                f"step {step}: episode terminated as crushed with neither pad on the "
                f"object (raw fingertip force {raw_N:.3f}N) -- the crush check is "
                "reading non-object contact again"
            )
            assert info["fingertip_force_N"] == 0.0, (
                f"step {step}: fingertip_force_N was {info['fingertip_force_N']:.3f}N "
                f"with neither pad on the object; only force through a pad actually "
                "touching the object counts as grip force"
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    assert max_off_object_force_N > fragile.crush_force_N, (
        f"never generated off-object fingertip force above crush_force_N "
        f"({max_off_object_force_N:.3f}N vs {fragile.crush_force_N}N), so this test "
        "never exercised the gate -- the arm probably didn't reach the tabletop"
    )
    LOGGER.info(
        "[5] OK: %.3fN of off-object fingertip force (>> crush_force_N=%.1fN) never "
        "counted as a crush.",
        max_off_object_force_N,
        fragile.crush_force_N,
    )


def assert_width_mass_set_is_accurate() -> None:
    """[4] generate_width_mass_objects.py's outputs reproduce their target width/mass."""
    for params in generate_objects():
        # generate_objects() itself asserts width/mass reproduction per-object (see
        # scripts/generate_width_mass_objects.py::_make_object) -- re-run it here so
        # this test fails loudly if that invariant ever regresses.
        assert params.rest_width_mm > 0 and params.mass_g > 0
    LOGGER.info("[4] OK: all 5 width/mass-matched objects reproduce their targets.")


def main() -> int:
    setup_logging()
    assert_baseline_obs_includes_force()
    assert_force_reading_is_genuine()
    assert_crush_terminates()
    assert_crush_ignores_non_object_contact()
    assert_width_mass_set_is_accurate()
    LOGGER.info("FORCE SENSOR TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
