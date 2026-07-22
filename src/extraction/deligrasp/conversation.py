"""
Thinker -> Coder chain, mirroring magpie/prompt_planner/conversation.py.

Stage 1 (thinker): natural-language instruction -> structured grasp description
                    with inferred mass / friction / spring constant / aperture.
Stage 2 (coder):    structured description -> executable gripper program.
"""

import re
from . import process_code


def _extract_description(response: str) -> str:
    """Pull the text between [start of description] and [end of description]."""
    try:
        body = re.split("end of description",
                        re.split("start of description", response, flags=re.IGNORECASE)[1],
                        flags=re.IGNORECASE)[0]
        return body.strip("[").strip("]").strip().strip("`")
    except Exception:
        return response


class Conversation:
    def __init__(self, backend, verbose=True):
        self.backend = backend
        self.verbose = verbose
        self.last_description = None
        self.last_code = None

    def plan(self, user_command: str):
        """Run the full pipeline. Returns (description, code)."""
        raw_desc = self.backend.thinker(user_command)
        description = _extract_description(raw_desc)
        if self.verbose:
            print("=" * 70)
            print("THINKER (inferred object properties + grasp params):")
            print(description)
        code = process_code.process_code_block(self.backend.coder(description))
        if self.verbose:
            print("-" * 70)
            print("CODER (generated grasp program):")
            print(code)
            print("=" * 70)
        self.last_description, self.last_code = description, code
        return description, code
