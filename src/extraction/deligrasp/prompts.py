"""
The actual DeliGrasp thinker/coder prompts, copied verbatim from
magpie/prompt_planner/prompts/mp_prompt_tc_phys.py so the reproduction uses the
paper's real prompting. Also holds the "LLM prior" the offline MockBackend uses
to stand in for GPT-4 when no API key is available.
"""

PROMPT_THINKER = """
Control a robot gripper with force control and contact information.
The gripper's parameters can be adjusted corresponding to the type of object that it is trying to grasp as well as the kind of grasp it is attempting to perform.
The gripper has a measurable max force of 16N and min force of 0.15N, a maximum aperture of 105mm and a minimum aperture of 1mm.

Some grasps may be incomplete, intended for observing force information about a given object.
Describe the grasp strategy using the following form:

[start of description]
* This {CHOICE: [is, is not]} a new grasp.
* In accordance with the user instruction, this grasp should be [GRASP_DESCRIPTION: <str>].
* This is a {CHOICE: [complete, incomplete]} grasp.
* This grasp {CHOICE: [does, does not]} contain multiple grasps.
* This object has more mass than [example object:  <str>], with mass of [PNUM: 0.0] g, and less mass than [example object:  <str>], with mass of [PNUM: 0.0] g
* Typically, this object's mass is approximately [PNUM: 0.0] g, which is between these two masses.
* Because the user specified that [OBJECT_DESCRIPTION: <str>], compared to typical, this object has a {CHOICE: [greater, lesser, similar]} mass of [PNUM: 0.0] grams.
* This grasp is for an object with {CHOICE: [high, medium, low]} compliance.
* The object has an approximate spring constant of [PNUM: 0.0] Newtons per meter.
* The gripper and object have an approximate friction coefficient of [PNUM: 0.0]
* This grasp should set the goal aperture to [PNUM: 0.0] mm.
* If the gripper slips, this grasp should close an additional [PNUM: 0.0] mm.
* If the gripper slips, this grasp should increase the output force by [PNUM: 0.0] Newtons.
* [optional] Because of [GRASP_DESCRIPTION: <str>], this grasp sets the force to be {CHOICE: [lower, higher]} than the default minimum grasp force.
[end of description]

Rules:

1. If you see phrases like {NUM: default_value}, replace the entire phrase with a numerical value.
2. If you see phrases like {CHOICE: [choice1, choice2, ...]}, replace the entire phrase with one of the choices listed.
3. Using knowledge of the object and how compliant it is, estimate the spring constant of the object. This can range broadly from 20 N/m for a very soft object to 2000 N/m for a very stiff object.
4. The increase in gripper output force is the maximum value of (0.05 N, or the product of the estimated aperture closure, the spring constant of the object, and a damping constant 0.1: (k*additional_closure*0.0001)).
5. Provide the full description. Always start with [start of description] and end with [end of description].
"""

PROMPT_CODER = """
We have a description of a gripper's motion and force sensing and we want you to turn that into the corresponding program with the following class functions of the gripper.
The gripper has a measurable max force of 16N and min force of 0.15N, a maximum aperture of 105mm and a minimum aperture of 1mm.

def get_aperture(finger='both')
def set_goal_aperture(aperture, finger='both', record_load=False)
def set_compliance(margin, flexibility, finger='both')
def set_force(force, finger='both')
def deligrasp(goal_aperture, initial_force, additional_closure, additional_force, complete_grasp)

Example answer code:
```
from magpie.gripper import Gripper # must import the gripper class
G = Gripper()
import numpy as np

goal_aperture = {PNUM: goal_aperture}
complete_grasp = {CHOICE: [True, False]}
# Initial force. Convert mass (g) to (kg). The default value of object weight / friction coefficient.
initial_force = ({PNUM: mass} * 9.81) / ({PNUM: mu} * 1000)
additional_closure = {PNUM: additional_closure}
# Additional force increase = spring constant * additional_closure * damping (0.1).
additional_force = np.max([0.01, additional_closure * {PNUM: spring_constant} * 0.0001])

G.set_goal_aperture(goal_aperture + additional_closure * 2, finger='both', record_load=False)
G.set_compliance(1, 3, finger='both')
G.set_force(initial_force, 'both')
G.deligrasp(goal_aperture, initial_force, additional_closure, additional_force, complete=complete_grasp, debug=True)
```

Rules:
1. Only use the functions listed above. Do not invent new functions.
2. The only allowed library is numpy.
3. Import the gripper class and create a Gripper at the beginning.
"""


# ---------------------------------------------------------------------------
# Offline stand-in for GPT-4. These are the *model's beliefs* about each object
# (deliberately kept separate from the ground truth in objects.py). Small,
# realistic estimation error is baked in so the harness exercises the same
# success/failure logic a real LLM would.
#   (est_mass_g, est_friction, est_spring_Npm, est_width_mm)
LLM_PRIORS = {
    "glass_bottle": (230,  0.30, 3800, 48),
    "steel_bolt":   (45,   0.65, 5500, 15),
    "ceramic_mug":  (300,  0.45, 3200, 78),
    "rice_bag":     (950,  0.75,  350, 42),
    "raw_egg":      (60,   0.28, 2800, 38),
    "brick":        (1600, 0.65, 7500, 42),
    # generic fallback used for unknown objects
    "_default":     (100, 0.50, 1000, 60),
}


def prior_for(object_name: str):
    key = object_name.lower().replace(" ", "_")
    for k in LLM_PRIORS:
        if k != "_default" and k in key:
            return k, LLM_PRIORS[k]
    return key, LLM_PRIORS["_default"]
