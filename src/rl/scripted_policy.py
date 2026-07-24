"""A scripted, non-learned reach/descend/grasp/lift heuristic for robosuite's ``Lift`` task.

Not an RL policy -- pure proportional control on ``cube_pos``/``robot0_eef_pos`` read
from the raw robosuite obs dict (``RobosuiteLiftEnv.last_obs_dict``). Generic across
objects: it never reads ``ObjectParams``, only the cube's and end-effector's world
positions, both present regardless of shape/size/mass/density.

Used two ways elsewhere in this repo:

- ``rl/sparse_seed.py`` runs it to find real successes and seed SAC's replay buffer
  when training with the sparse/"fixed" reward (``EnvConfig.reward_shaping=False``).
- ``scripts/scripted_rollout.py`` runs it standalone -- no SAC, no training, no
  ``ObjectParams`` -- to check whether the heuristic alone can lift the cube.
"""

from __future__ import annotations

import numpy as np

_HOVER_HEIGHT_M = 0.08
_XY_TOL_M = 0.015
_DESCEND_TOL_M = 0.012
_GRASP_HOLD_STEPS = 10
_GAIN = 4.0

#: 3 position deltas + >=1 gripper dim. Below this the heuristic has nowhere to put
#: its xyz/gripper commands -- callers should check and skip gracefully.
MIN_ACTION_DIM = 4


class ScriptedPickPolicy:
    """Proportional-control reach/descend/grasp/lift heuristic. Not a learned policy.

    Degrades gracefully (``act`` returns ``None``) if ``cube_pos``/``robot0_eef_pos``
    are missing from the obs dict, e.g. a task/robot this wasn't written against.
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
