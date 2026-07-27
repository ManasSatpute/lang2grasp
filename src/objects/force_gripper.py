"""A Panda gripper variant with a genuine per-fingertip MuJoCo force sensor.

``rl/env.py``'s grip-force-aware reward shaping used to *estimate* per-finger contact
force from gripper aperture (a spring-compression model,
``ObjectParams.reaction_force_N``) -- never a real sensor reading, and only ever wired
up when ``EnvConfig.object`` was set, so a force-conditioned policy's apparent edge was
partly just "has force info the generic baseline never saw."

This module replaces that estimate with a real MuJoCo ``<force>`` sensor on each
fingertip pad. robosuite's own ``Wipe`` task already reads exactly this kind of
sensor for a contact-force reward/termination (``environments/manipulation/wipe.py``
via ``Robot.ee_force`` -> ``Robot.get_sensor_measurement`` -> the Panda gripper's
wrist ``force_ee`` sensor at its ``ft_frame`` site) -- confirmed against the installed
robosuite 1.5.x source. MuJoCo's site force/torque sensor reports the interaction
force (``cfrc_int``) on the site's *body*, computed regardless of whether that body
has its own joint, so it works just as well on the fingertip pad bodies
(``finger_joint1_tip`` / ``finger_joint2_tip`` in the stock ``panda_gripper.xml``),
which are rigid, jointless children of the finger bodies -- the same structural
situation as the wrist ``ft_frame`` site or ``wiping_gripper.xml``'s per-corner touch
sites.

robosuite hardcodes the stock XML path in ``PandaGripperBase.__init__`` with no
constructor hook to point elsewhere, so -- same vendoring rationale as
``lift_object_task.py``'s ``ParamLift`` -- this module derives a modified copy of the
installed ``panda_gripper.xml`` (two new ``<site>``s + two new ``<force>`` sensors,
one pair per pad) at import time, from whatever robosuite version is actually
installed, rather than shipping a static XML that could drift from it. Re-diff
against the installed ``panda_gripper.xml`` before raising the ``robosuite<1.6`` pin
in ``requirements.txt``.
"""

from __future__ import annotations

import functools
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import robosuite
from robosuite.models.grippers import register_gripper
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.panda_gripper import PandaGripper
from robosuite.utils.mjcf_utils import xml_path_completion

#: (tip body name, pad-local site pos, site name, sensor name) -- pos values copied
#: from each pad's own collision geom (``finger1_pad_collision`` /
#: ``finger2_pad_collision``) in the stock XML, so the sensor site sits centred on
#: the pad rather than at the tip body's origin.
_PAD_SENSORS: tuple[tuple[str, str, str, str], ...] = (
    ("finger_joint1_tip", "0 -0.005 -0.015", "left_pad_force_site", "left_pad_force"),
    ("finger_joint2_tip", "0 0.005 -0.015", "right_pad_force_site", "right_pad_force"),
)

#: Raw (un-prefixed) sensor names added by this module -- see ``PandaGripperForce.
#: _important_sensors``. Exposed so ``rl/env.py`` doesn't have to repeat the strings.
PAD_FORCE_SENSOR_NAMES: tuple[str, str] = ("left_pad_force", "right_pad_force")


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent / "_generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path() -> Path:
    # Version-qualified: a SLURM array job's tasks (each its own process, possibly on
    # different compute nodes) all import this module against the same NFS/GPFS-shared
    # repo checkout at roughly the same moment, so this file has many more concurrent
    # writers than a single local run does. Naming it after robosuite's own version
    # means (a) an already-built file for the currently-installed version is simply
    # reused (see the exists() check below) instead of every task re-deriving and
    # re-writing it, and (b) a conda env upgrade/downgrade across job submissions can
    # never load a stale/incompatible derived XML left over from a different version.
    safe_version = re.sub(r"[^A-Za-z0-9.]", "_", robosuite.__version__)
    return _cache_dir() / f"panda_gripper_force_{safe_version}.xml"


@functools.lru_cache(maxsize=1)
def build_panda_gripper_force_xml() -> str:
    """Return the path to a derived ``panda_gripper.xml`` with per-pad force sensors.

    Built once per installed robosuite version (cached in-process after that -- a
    single process may build several ``PandaGripperForce`` instances, e.g. train +
    eval envs, and the transform is deterministic) from whatever ``panda_gripper.xml``
    that robosuite version ships. Mesh ``file=`` paths are absolutised here so the
    derived file can live outside robosuite's own ``grippers/`` asset directory
    without breaking mesh loading -- this replicates what
    ``robosuite.models.base.MujocoXML.resolve_asset_dependency`` does automatically
    for a fragment's *original* directory, just done ahead of time for a directory
    the derived file doesn't actually live in.

    Many processes across many hosts can call this concurrently -- e.g. a SLURM
    array job's tasks, each spawning several ``SubprocVecEnv`` workers, all pointed at
    one shared repo checkout. Two things make that safe: the write is atomic (unique
    temp file + ``os.replace``, so a partial write is never read), and an existing
    cache file for the current robosuite version is reused as-is rather than
    regenerated, so in the steady state only the first caller on the whole cluster
    actually writes it.
    """
    out_path = _cache_path()
    if out_path.exists():
        return str(out_path)

    src_path = Path(xml_path_completion("grippers/panda_gripper.xml"))
    src_dir = src_path.parent

    tree = ET.parse(src_path)
    root = tree.getroot()

    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if file_attr and not os.path.isabs(file_attr):
            mesh.set("file", str((src_dir / file_attr).resolve()))

    sensor_el = root.find("sensor")
    for tip_body_name, pad_pos, site_name, sensor_name in _PAD_SENSORS:
        tip_body = root.find(f".//body[@name='{tip_body_name}']")
        if tip_body is None:
            raise RuntimeError(
                f"panda_gripper.xml (robosuite {robosuite.__version__}) has no body "
                f"named {tip_body_name!r}; robosuite's gripper XML changed shape -- "
                "re-diff force_gripper.py's assumptions against the installed "
                "version before raising the robosuite pin."
            )
        ET.SubElement(
            tip_body,
            "site",
            name=site_name,
            pos=pad_pos,
            size="0.003",
            type="sphere",
            group="1",
            rgba="1 0 0 0.3",
        )
        ET.SubElement(sensor_el, "force", name=sensor_name, site=site_name)

    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".xml.tmp")
    os.close(fd)
    try:
        tree.write(tmp_name)
        os.replace(tmp_name, out_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return str(out_path)


@register_gripper
class PandaGripperForce(PandaGripper):
    """``PandaGripper`` with two extra per-pad 3-axis force sensors.

    Registered under its class name (``register_gripper``), so
    ``suite.make(..., gripper_types="PandaGripperForce")`` resolves it like any
    built-in gripper. Action mapping, speed, dof, and important geoms are all
    inherited from ``PandaGripper`` unchanged -- only the XML (sensors added) and
    ``_important_sensors`` (new sensor names registered) differ.
    """

    def __init__(self, idn: int = 0) -> None:
        # Skip PandaGripperBase.__init__, which hardcodes the stock XML path -- go
        # straight to GripperModel.__init__ with our derived path instead.
        GripperModel.__init__(self, build_panda_gripper_force_xml(), idn=idn)

    @property
    def _important_sensors(self) -> dict[str, str]:
        sensors = dict(super()._important_sensors)
        sensors.update(zip(PAD_FORCE_SENSOR_NAMES, PAD_FORCE_SENSOR_NAMES))
        return sensors
