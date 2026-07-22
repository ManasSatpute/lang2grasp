"""A ``robosuite.Lift`` variant whose liftable object is built from :class:`ObjectParams`.

robosuite's ``Lift._load_model`` hardcodes ``self.cube = BoxObject(...)`` with no
constructor hook to swap it -- confirmed against the installed robosuite 1.5.1 source
(``robosuite/environments/manipulation/lift.py``). Every other ``Lift`` method
(``reward``, ``_check_success``, ``_setup_observables``, ``visualize``,
``_reset_internal``) references ``self.cube`` generically, so the only method that
needs overriding is ``_load_model`` -- and that means **vendoring its body**, since
there's no smaller extension point. This is pinned to the ``robosuite<1.6`` requirement
in ``requirements.txt``; re-diff this method against ``Lift._load_model`` before
raising that pin (same spirit as ``env.py``'s ``_load_controller_config`` handling the
1.4/1.5 controller-config API split).

robosuite auto-registers any subclass of its env base class by class name (see
``robosuite.environments.base.EnvMeta``), so importing this module is enough to make
``suite.make("ParamLift", ...)`` resolve -- no manual registry call needed.
"""

from __future__ import annotations

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject, MujocoObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from objects.object_params import ObjectParams

_BUILDERS = {"box": BoxObject, "cylinder": CylinderObject, "ball": BallObject}


def build_mujoco_object(params: ObjectParams) -> MujocoObject:
    """Construct the robosuite primitive object described by ``params``.

    Named ``"cube"`` regardless of shape: that's the name every untouched ``Lift``
    method (``reward``, ``_check_success``, ...) looks up on ``self``.
    """
    cls = _BUILDERS[params.shape]
    return cls(
        name="cube",
        size=params.size,
        density=params.density,
        friction=list(params.friction),
        rgba=list(params.rgba),
    )


class ParamLift(Lift):
    """``Lift`` with the cube replaced by an object built from ``object_params``."""

    def __init__(self, *args, object_params: ObjectParams, **kwargs) -> None:
        self.object_params = object_params
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        # Vendored from Lift._load_model (robosuite 1.5.1): identical arena setup,
        # only the cube construction differs. `super(Lift, self)` -- not `super()` --
        # skips straight to Lift's parent, since Lift's own _load_model is exactly
        # what we're replacing.
        super(Lift, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Stock Lift's cube uses a CustomMaterial texture; ours is colored via a flat
        # rgba on the geom instead (see build_mujoco_object), so no material here.
        self.cube = build_mujoco_object(self.object_params)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )
