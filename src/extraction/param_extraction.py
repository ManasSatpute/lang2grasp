"""Public entry point: text prompt -> validated :class:`ObjectParams`.

Thin orchestration over `llm_backends.ParamBackend`: call the backend, retry
once on malformed/incomplete JSON (backends occasionally wrap output in
markdown fences or drop a field), then hand off to `ObjectParams.__post_init__`
for the one place validation and clamping actually happen.
"""

from __future__ import annotations

import logging

from extraction.llm_backends import ParamBackend
from objects.object_params import ObjectParams

LOGGER = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("shape", "size", "density", "friction")


def _validate_fields(fields: dict) -> None:
    missing = [f for f in _REQUIRED_FIELDS if f not in fields]
    if missing:
        raise ValueError(f"Backend response missing field(s): {missing}")


def extract_object_params(name: str, prompt: str, backend: ParamBackend) -> ObjectParams:
    """Run ``backend`` on ``prompt`` and return a validated :class:`ObjectParams`.

    Retries once if the backend's response is missing a required field --
    cheap insurance against an LLM dropping a key on a bad turn.
    """
    try:
        fields = backend.extract(prompt)
        _validate_fields(fields)
    except (ValueError, KeyError) as exc:
        LOGGER.warning("Malformed extraction response (%s); retrying once.", exc)
        fields = backend.extract(prompt)
        _validate_fields(fields)

    fields = dict(fields)
    fields.pop("name", None)  # `name` is supplied by the caller, not the backend
    return ObjectParams(name=name, **fields)
