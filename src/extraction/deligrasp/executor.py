"""
Execute LLM-generated grasp code against the simulator.

The real system writes the code to a temp file and runs it in a subprocess with
stdout captured (ConfirmationSafeExecutor). Here we execute in-process for
clarity and to inject the active object, but we still (a) rewrite the
`from magpie.gripper import Gripper` import to the simulator and (b) capture
stdout, so behaviour matches the original: the printed grasp_log is recovered.
"""

import io
import contextlib
from . import gripper as sim_gripper


def run(code: str, obj):
    """Execute `code` with `obj` as the object being grasped.

    Returns dict with final aperture/applied force, the structured grasp log,
    the peak contact force (for crush evaluation), and captured stdout.
    """
    # Point the generated `from magpie.gripper import Gripper` at the simulator.
    patched = code.replace("from magpie.gripper import Gripper",
                           "from deligrasp.gripper import Gripper")

    sim_gripper.ACTIVE_OBJECT = obj
    sim_gripper.LAST_GRASP_LOG = []
    # track the gripper instance the code creates so we can read peak force
    created = {}
    orig_init = sim_gripper.Gripper.__init__

    def tracking_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        created["g"] = self

    sim_gripper.Gripper.__init__ = tracking_init
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(patched, "<llm_grasp_code>", "exec"), {})
    finally:
        sim_gripper.Gripper.__init__ = orig_init

    g = created.get("g")
    return {
        "final_aperture_mm": g.aperture if g else None,
        "applied_force_N": g.applied_force if g else None,
        "peak_contact_force_N": g.peak_contact_force if g else 0.0,
        "grasp_log": list(sim_gripper.LAST_GRASP_LOG),
        "stdout": buf.getvalue(),
    }
