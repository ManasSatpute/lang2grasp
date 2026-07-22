"""
LLM backends.

`OpenAIBackend` reproduces the original conversation.py call (temperature 0.3,
retry if the model left {CHOICE}/{NUM} placeholders unfilled).

`MockBackend` is a deterministic, offline stand-in that fills the thinker
template from LLM_PRIORS and emits coder output in exactly the format the real
coder produces, so the entire pipeline is runnable and testable with no API key.
Swapping backends changes nothing else in the pipeline.
"""

import time
import numpy as np
from . import prompts


class LLMBackend:
    def thinker(self, user_command: str) -> str:
        raise NotImplementedError

    def coder(self, thinker_description: str) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """Real GPT backend. Requires OPENAI_API_KEY and `pip install openai`."""

    def __init__(self, model="gpt-4-turbo", temperature=0.3):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def _call(self, system_prompt, user_content):
        reset_triggers = ["CHOICE", "NUM"]
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                text = completion.choices[0].message.content
                if not any(t in text for t in reset_triggers):
                    return text
            except Exception as e:  # pylint: disable=broad-exception-caught
                print("OpenAI call issue, retrying...", e)
                time.sleep(5)

    def thinker(self, user_command):
        return self._call(prompts.PROMPT_THINKER,
                          user_command + " Make sure to ignore irrelevant options.")

    def coder(self, thinker_description):
        return self._call(prompts.PROMPT_CODER, thinker_description)


class MockBackend(LLMBackend):
    """Offline deterministic backend approximating GPT-4's structured output."""

    def __init__(self, damping=0.1):
        self.damping = damping

    def thinker(self, user_command):
        key, (mass, mu, k, width) = prompts.prior_for(user_command)
        # aperture closure to attempt on slip: small relative to width, min 1mm
        closure = max(1.0, round(width * 0.03, 1))
        add_force = max(0.05, k * closure * 0.0001)
        goal = round(width, 1)  # start contact at estimated width, then adapt
        compliance = "low" if k > 1500 else ("medium" if k > 400 else "high")
        return f"""[start of description]
* This is a new grasp.
* In accordance with the user instruction, this grasp should be a stable pick of the {key}.
* This is a complete grasp.
* This grasp does not contain multiple grasps.
* This object has more mass than a feather, with mass of 0.5 g, and less mass than a brick, with mass of 2000 g
* Typically, this object's mass is approximately {mass} g, which is between these two masses.
* Because the user specified a {key}, compared to typical, this object has a similar mass of {mass} grams.
* This grasp is for an object with {compliance} compliance.
* The object has an approximate spring constant of {k} Newtons per meter.
* The gripper and object have an approximate friction coefficient of {mu}
* This grasp should set the goal aperture to {goal} mm.
* If the gripper slips, this grasp should close an additional {closure} mm.
* If the gripper slips, this grasp should increase the output force by {round(add_force, 4)} Newtons.
[end of description]"""

    def coder(self, thinker_description):
        import re

        def num(pattern, default):
            m = re.search(pattern, thinker_description)
            return float(m.group(1)) if m else default

        mass = num(r"mass of ([\d.]+) grams", 100)
        k = num(r"spring constant of ([\d.]+)", 1000)
        mu = num(r"friction coefficient of ([\d.]+)", 0.5)
        goal = num(r"goal aperture to ([\d.]+) mm", 60)
        closure = num(r"close an additional ([\d.]+) mm", 2)

        code = f"""from magpie.gripper import Gripper
G = Gripper()
import numpy as np

goal_aperture = {goal}
complete_grasp = True
# Initial force = object weight / friction coefficient (mass g -> kg).
initial_force = ({mass} * 9.81) / ({mu} * 1000)
additional_closure = {closure}
additional_force = np.max([0.01, additional_closure * {k} * 0.0001])

G.set_goal_aperture(goal_aperture + additional_closure * 2, finger='both', record_load=False)
G.set_compliance(1, 3, finger='both')
G.set_force(initial_force, 'both')
G.deligrasp(goal_aperture, initial_force, additional_closure, additional_force, complete=complete_grasp, debug=True)
"""
        return "```python\n" + code + "```"
