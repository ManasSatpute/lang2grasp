"""A ``robosuite.Lift`` variant whose liftable object is built from :class:`ObjectParams`.

robosuite's ``Lift._load_model`` hardcodes the cube with no hook to swap it, so
``_load_model`` below is a vendored copy with only the object construction changed.
Every other ``Lift`` method references ``self.cube`` generically -- except
``_check_success``, whose height threshold is calibrated for the stock 4cm cube and is
overridden here (see :meth:`ParamLift._check_success`). Pinned to ``robosuite<1.6`` in
``requirements.txt`` -- re-diff both methods against ``Lift`` before raising that pin.

robosuite auto-registers any subclass of its env base class by class name, so
importing this module is enough to make ``suite.make("ParamLift", ...)`` resolve.
"""

from __future__ import annotations

from collections.abc import Callable

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject, MujocoObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from objects.object_params import ObjectParams

_BUILDERS = {"box": BoxObject, "cylinder": CylinderObject, "ball": BallObject}

#: How far above its own resting height the object must rise to count as lifted (m).
#: robosuite's `Lift` hardcodes an *absolute* `table_z + 0.04`, which only reads as
#: "lift by 2cm" for its stock cube (half-extent 0.02). 0.02 here reproduces that
#: difficulty exactly for the stock cube while staying meaningful for every other size.
LIFT_MARGIN_M = 0.02


def resting_center_height(params: ObjectParams) -> float:
    """Height (m) of the object's body centre above the table when it sits at rest.

    Equal to its half-extent along z: `size[2]` for a box, `size[1]` (half-height) for
    an upright cylinder, `size[0]` (radius) for a ball.
    """
    if params.shape == "box":
        return params.size[2]
    if params.shape == "cylinder":
        return params.size[1]
    return params.size[0]  # ball

#: Either a fixed object (the per-object-specialist path) or a zero-arg callable
#: drawing a fresh one (the domain-randomization path, e.g.
#: `objects.object_params.sample_object_params`) -- see `ParamLift.__init__`.
ObjectParamsSource = ObjectParams | Callable[[], ObjectParams]


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
    """``Lift`` with the cube replaced by an object built from ``object_params``.

    When ``object_params`` is a callable, ``_load_model`` re-invokes it every call --
    e.g. one fresh sample per episode for domain randomization (``EnvConfig.
    randomize_object`` in ``rl/env.py``). ``self.object_params`` always reflects
    whichever instance is currently loaded; callers should read it back after
    ``reset()``, not the constructor argument.
    """

    def __init__(self, *args, object_params: ObjectParamsSource, **kwargs) -> None:
        self._object_params_source = object_params
        self.object_params = object_params() if callable(object_params) else object_params
        super().__init__(*args, **kwargs)

    def _load_model(self) -> None:
        # `super(Lift, self)` skips straight to Lift's parent, since Lift's own
        # _load_model is exactly what this method replaces.
        super(Lift, self)._load_model()

        if callable(self._object_params_source):
            self.object_params = self._object_params_source()

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

    def _check_success(self) -> bool:
        """Has the object been lifted clear of the table?

        ``Lift._check_success`` tests ``body_z > table_z + 0.04``, an absolute threshold
        calibrated for its stock 4cm cube. Applied to an arbitrary ``ObjectParams`` it
        breaks both ways: anything taller than 8cm (a 18cm ``glass_bottle``, a 9cm
        ``ceramic_mug``, four of the five ``width_mass_set`` cylinders) already satisfies
        it sitting *untouched* on the table, while a short object has to be lifted
        further than the cube did. Measure the lift relative to the object's own resting
        height instead, so "lifted" means the same thing for every object.
        """
        object_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        resting_height = resting_center_height(self.object_params)
        return bool(object_height > table_height + resting_height + LIFT_MARGIN_M)
