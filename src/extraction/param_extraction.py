"""Public entry point: text prompt -> validated :class:`ObjectParams`.

Thin orchestration over `llm_backends.ParamBackend`: call the backend, retry
once on malformed/incomplete JSON (backends occasionally wrap output in
markdown fences, drop a field, or -- backends without server-side schema
enforcement, e.g. GroqBackend's `json_object` mode -- put the wrong *type* in a
field), then hand off to `ObjectParams.__post_init__` for the one place
validation and clamping actually happen.
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


def _build(name: str, fields: dict) -> ObjectParams:
    _validate_fields(fields)
    fields = dict(fields)
    fields.pop("name", None)  # `name` is supplied by the caller, not the backend
    return ObjectParams(name=name, **fields)


def extract_object_params(name: str, prompt: str, backend: ParamBackend) -> ObjectParams:
    """Run ``backend`` on ``prompt`` and return a validated :class:`ObjectParams`.

    Retries once on a malformed response -- a missing field, or a field with the
    wrong type/value (surfaces as `ObjectParams.__post_init__` raising `ValueError`
    or `TypeError`) -- cheap insurance against an LLM going off-schema on a bad turn.
    """
    try:
        fields = backend.extract(prompt)
        return _build(name, fields)
    except (ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Malformed extraction response (%s); retrying once.", exc)
        fields = backend.extract(prompt)
        return _build(name, fields)
