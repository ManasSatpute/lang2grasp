"""Extract executable code from an LLM response. Ported from the original
magpie/prompt_planner/process_code.py."""

import re


def _fix_code(code_str: str) -> str:
    if "np." in code_str and "import numpy as np" not in code_str:
        code_str = "import numpy as np\n" + code_str
    return code_str


def process_code_block(text: str) -> str:
    matches = re.findall(r"```(python)?\n?([\s\S]*?)```", text)
    code = "\n".join(m[1] for m in matches) if matches else text
    return _fix_code(code)
