"""A Panda gripper variant with a genuine per-fingertip MuJoCo force sensor.

Adds a real MuJoCo ``<force>`` sensor on each fingertip pad, so `rl/env.py`'s
grip-force reward shaping reads an actual contact force instead of estimating one
from gripper aperture. robosuite hardcodes the stock gripper XML path with no hook to
swap it, so this module derives a modified copy of the installed ``panda_gripper.xml``
(two new sites + force sensors) at import time. Re-diff against the installed XML
before raising the ``robosuite<1.6`` pin in ``requirements.txt``.
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

#: (tip body name, pad-local site pos, site name, sensor name). Site positions are
#: centred on each pad's own collision geom in the stock XML.
_PAD_SENSORS: tuple[tuple[str, str, str, str], ...] = (
    ("finger_joint1_tip", "0 -0.005 -0.015", "left_pad_force_site", "left_pad_force"),
    ("finger_joint2_tip", "0 0.005 -0.015", "right_pad_force_site", "right_pad_force"),
)

#: Sensor names added by this module, exposed so `rl/env.py` doesn't repeat the strings.
PAD_FORCE_SENSOR_NAMES: tuple[str, str] = ("left_pad_force", "right_pad_force")


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent / "_generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path() -> Path:
    # Version-qualified so a cache built under one installed robosuite version is
    # never reused (or clobbered) by a different one.
    safe_version = re.sub(r"[^A-Za-z0-9.]", "_", robosuite.__version__)
    return _cache_dir() / f"panda_gripper_force_{safe_version}.xml"


@functools.lru_cache(maxsize=1)
def build_panda_gripper_force_xml() -> str:
    """Return the path to a derived ``panda_gripper.xml`` with per-pad force sensors.

    Built once per installed robosuite version and cached to disk; mesh ``file=``
    paths are absolutised so the derived file works outside robosuite's own asset
    directory. The write is atomic (temp file + ``os.replace``) and an existing cache
    is reused as-is, so concurrent callers (e.g. several `SubprocVecEnv` workers
    across a SLURM array job, sharing one repo checkout) are safe.
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
