"""
Simulated force-sensing parallel-jaw gripper.

This is a drop-in replacement for `magpie.gripper.Gripper`. It exposes the same
high-level API the LLM-generated code calls (get_aperture, set_goal_aperture,
set_compliance, set_force, deligrasp, poke) but instead of driving Dynamixel
servos and reading load registers, it simulates the contact force produced by
compressing a spring-like object.

The object being grasped is injected via the module global ACTIVE_OBJECT before
the generated code is executed -- this mirrors the real world where the gripper
simply grasps whatever single object is in front of it.

Contact model (per finger):
    reaction(aperture) = k * max(0, rest_width - aperture) / 1000      [N]
    measured_force     = min(force_limit, reaction)                    [N]
The gripper closes toward a commanded aperture but stalls when the object's
reaction reaches the force limit, so measured_force is what a load-cell finger
would read.
"""

import numpy as np
import time

# Injected by the executor before running LLM code. None -> free-space (no object).
ACTIVE_OBJECT = None
# Structured trajectory of the most recent deligrasp() call, for analysis.
LAST_GRASP_LOG = []


class Gripper:
    MAX_FORCE_N = 16.0
    MIN_FORCE_N = 0.15
    MAX_APERTURE_MM = 105.0
    MIN_APERTURE_MM = 1.0
    SENSING_FLOOR_N = 0.15  # cannot resolve contact below this

    def __init__(self, servoport=None, debug=False):
        self.obj = ACTIVE_OBJECT
        self.debug = debug
        self.aperture = self.MAX_APERTURE_MM      # start fully open
        self.force_limit = self.MIN_FORCE_N       # current per-finger force cap
        self.compliance_margin = 1
        self.compliance_slope = 3
        # bookkeeping the real gripper exposes / the VLA code reads
        self.applied_force = self.MIN_FORCE_N
        self.recorded_contact_force = 0.0
        self.peak_contact_force = 0.0             # for crush evaluation

    # ----- physics -------------------------------------------------------
    def _reaction(self, aperture_mm):
        """Object restoring force (N) at a given aperture, ignoring force limit."""
        if self.obj is None:
            return 0.0
        compression_m = max(0.0, (self.obj.rest_width_mm - aperture_mm)) / 1000.0
        return self.obj.spring_Npm * compression_m

    def _settle(self, goal_mm, force_limit_N):
        """Close toward goal_mm, force-limited. Returns measured per-finger force.

        If the object's reaction would exceed the force limit before reaching the
        goal, the finger stalls at the equilibrium aperture.
        """
        goal_mm = float(np.clip(goal_mm, self.MIN_APERTURE_MM, self.MAX_APERTURE_MM))
        if self.obj is None or goal_mm >= self.obj.rest_width_mm:
            self.aperture = goal_mm
            measured = 0.0
        else:
            reaction_at_goal = self._reaction(goal_mm)
            if reaction_at_goal <= force_limit_N:
                self.aperture = goal_mm                      # reached goal
                measured = reaction_at_goal
            else:
                # stall where reaction == force limit
                stall = self.obj.rest_width_mm - force_limit_N * 1000.0 / self.obj.spring_Npm
                self.aperture = max(self.MIN_APERTURE_MM, stall)
                measured = force_limit_N
        self.recorded_contact_force = measured
        self.peak_contact_force = max(self.peak_contact_force, measured)
        return measured

    # ----- low-level API used by generated code --------------------------
    def get_aperture(self, finger="both"):
        return self.aperture

    def set_force(self, force, finger="both"):
        self.force_limit = float(np.clip(force, self.MIN_FORCE_N, self.MAX_FORCE_N))
        self.applied_force = self.force_limit

    def set_compliance(self, margin, flexibility, finger="both"):
        self.compliance_margin = margin
        self.compliance_slope = flexibility

    def set_goal_aperture(self, aperture, finger="both", record_load=False):
        measured = self._settle(aperture, self.force_limit)
        # emulate the (2, n) position-load array the real gripper returns
        pos = np.array([self.aperture, self.aperture])
        load = np.array([measured, measured])
        return np.array([pos, load]), measured, measured

    def poke(self, direction, speed, aperture, debug=False):
        # not needed for grasping; provided so generated code never breaks
        return self.aperture

    def check_slip(self, pos_load, stop_force, finger="both"):
        """True if the target contact force was NOT reached (object slips)."""
        target_per_finger = max(self.SENSING_FLOOR_N, stop_force / 2.0)
        measured = self.recorded_contact_force
        slip = measured < target_per_finger
        return slip, measured, measured

    # ----- the DeliGrasp adaptive controller -----------------------------
    def deligrasp(self, goal_aperture, initial_force, additional_closure,
                  additional_force, complete=True, debug=False):
        """Close to goal, then adapt (close more + raise force) until the object
        no longer slips, i.e. until the measured contact force reaches the target.

        Faithful to magpie.gripper.Gripper.deligrasp: step-and-check on slip,
        logging a per-step trajectory that is printed to stdout.
        """
        global LAST_GRASP_LOG
        self.debug = debug
        self.peak_contact_force = 0.0
        grasp_log = []

        fc = initial_force                     # target contact force (total)
        self.set_force(initial_force)
        applied = initial_force
        goal = goal_aperture

        # initial grasp attempt
        _, measured, _ = self.set_goal_aperture(goal, finger="both", record_load=True)
        slip, measured, _ = self.check_slip(None, fc)

        t0 = time.time()
        grasp_log.append(dict(step=0, timestamp=t0, aperture=round(self.aperture, 3),
                              contact_force=round(measured, 4),
                              applied_force=round(applied, 4), k=0.0, slip=slip))

        k_est = self.obj.spring_Npm if self.obj else 0.0
        step = 1
        # loop guard so a bad plan (target unreachable) terminates
        while slip and self.aperture > self.MIN_APERTURE_MM and step < 500:
            goal -= additional_closure
            if measured > 0.10:                # only raise force once in contact
                applied = min(applied + additional_force, self.MAX_FORCE_N)
            self.set_force(applied)
            prev_ap = self.aperture
            _, measured, _ = self.set_goal_aperture(goal, finger="both", record_load=True)
            slip, measured, _ = self.check_slip(None, fc)
            k = measured * abs(self.aperture - prev_ap) * 1000.0
            grasp_log.append(dict(step=step, timestamp=time.time(),
                                  aperture=round(self.aperture, 3),
                                  contact_force=round(measured, 4),
                                  applied_force=round(applied, 4),
                                  k=round(k, 3), slip=slip))
            step += 1

        if complete:
            self.set_goal_aperture(self.aperture, finger="both")  # hold
        else:
            self.aperture = self.MAX_APERTURE_MM                   # release

        LAST_GRASP_LOG = grasp_log
        # IMPORTANT: printed so a subprocess executor could capture it (as in the
        # original system). Analysis here reads the structured LAST_GRASP_LOG.
        print(grasp_log)
        return self.aperture, applied, [k_est], grasp_log

    # ----- naive baselines (not LLM-driven) ------------------------------
    def naive_grasp(self, goal_aperture, force_per_finger):
        """Constant-force grasp: squeeze the object with a fixed force limit.

        Models a non-adaptive controller with no property inference. Closes past
        the object (toward min aperture) subject to the force limit.
        """
        self.peak_contact_force = 0.0
        self.set_force(force_per_finger)
        measured = self._settle(self.MIN_APERTURE_MM, self.force_limit)
        return self.aperture, self.force_limit, measured
