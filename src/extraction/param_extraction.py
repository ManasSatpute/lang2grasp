"""Public entry point: text prompt -> validated :class:`ObjectParams`.

Thin orchestration over `llm_backends.ParamBackend`: call the backend (optionally
several times -- see `extract_object_params`'s `samples`), retry a malformed/
incomplete response once (backends occasionally wrap output in markdown fences,
drop a field, or -- backends without server-side schema enforcement, e.g.
GroqBackend's `json_object` mode -- put the wrong *type* in a field), then hand off
to `ObjectParams.__post_init__` for the one place validation and clamping actually
happen.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
from collections import Counter

from extraction.llm_backends import ParamBackend
from objects.object_params import ObjectParams

LOGGER = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("shape", "size", "density", "friction")
#: Scalar numeric fields, aggregated by per-field median across samples.
_NUMERIC_SCALAR_FIELDS = ("density", "grip_force_min_N", "grip_force_max_N", "spring_Npm", "crush_force_N")
#: List-valued numeric fields, aggregated by elementwise median across samples.
_NUMERIC_LIST_FIELDS = ("friction", "rgba")


def _validate_fields(fields: dict) -> None:
    missing = [f for f in _REQUIRED_FIELDS if f not in fields]
    if missing:
        raise ValueError(f"Backend response missing field(s): {missing}")


def _build(name: str, fields: dict) -> ObjectParams:
    _validate_fields(fields)
    fields = dict(fields)
    fields.pop("name", None)  # `name` is supplied by the caller, not the backend
    return ObjectParams(name=name, **fields)


def _extract_once(prompt: str, backend: ParamBackend) -> dict:
    """One backend call, turned into a complete, validated, clamped fields dict via a
    throwaway :class:`ObjectParams` build -- so a caller aggregating multiple samples
    always sees the same, consistent shape (every field present, right type, in
    range) regardless of what the backend actually returned. Raises `ValueError`/
    `KeyError`/`TypeError` on a malformed response.
    """
    fields = backend.extract(prompt)
    probe = _build("_sample", fields)
    return {f.name: getattr(probe, f.name) for f in dataclasses.fields(ObjectParams) if f.name != "name"}


def _extract_with_retry(prompt: str, backend: ParamBackend) -> dict | None:
    """`_extract_once`, retried once on a malformed response. Returns ``None`` (after
    logging) if the retry is also malformed, so a multi-sample caller can drop this
    one sample instead of the whole run failing over a single bad turn.
    """
    try:
        return _extract_once(prompt, backend)
    except (ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Malformed extraction response (%s); retrying once.", exc)
        try:
            return _extract_once(prompt, backend)
        except (ValueError, KeyError, TypeError) as exc2:
            LOGGER.warning("Still malformed after retry (%s).", exc2)
            return None


def _majority(values: list) -> object:
    """Most common value; ties broken by first occurrence."""
    counts = Counter(values)
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def _aggregate(samples: list[dict]) -> dict:
    """Combine several validated fields dicts into one: median for numeric fields,
    majority vote for categorical ones. `size` is only averaged across samples that
    agree with the majority `shape` (different shapes mean differently-shaped `size`
    tuples, so mixing them doesn't make sense).
    """
    shape = _majority([s["shape"] for s in samples])
    by_shape = [s for s in samples if s["shape"] == shape]

    agg = {
        "shape": shape,
        "size": [statistics.median(vals) for vals in zip(*(s["size"] for s in by_shape))],
        "mass_class": _majority([s["mass_class"] for s in samples]),
        "fragile": _majority([s["fragile"] for s in samples]),
    }
    for field in _NUMERIC_SCALAR_FIELDS:
        agg[field] = statistics.median(s[field] for s in samples)
    for field in _NUMERIC_LIST_FIELDS:
        agg[field] = [statistics.median(vals) for vals in zip(*(s[field] for s in samples))]
    return agg


def _log_spread(name: str, samples: list[dict], aggregated: dict) -> None:
    """INFO-log which numeric fields actually disagreed across samples, and by how
    much -- visibility into how noisy a field currently is, not just its final value.
    """
    parts = [
        f"{field}={min(vals):.3g}-{max(vals):.3g} (median {aggregated[field]:.3g})"
        for field in _NUMERIC_SCALAR_FIELDS
        for vals in [[s[field] for s in samples]]
        if max(vals) != min(vals)
    ]
    if parts:
        LOGGER.info("%s: sample spread -- %s", name, "; ".join(parts))
    shapes = {s["shape"] for s in samples}
    if len(shapes) > 1:
        LOGGER.info("%s: shape disagreement across samples: %s (took majority: %s)", name, sorted(shapes), aggregated["shape"])


def extract_object_params(name: str, prompt: str, backend: ParamBackend, samples: int = 1) -> ObjectParams:
    """Run ``backend`` on ``prompt`` ``samples`` time(s) and return one validated
    :class:`ObjectParams`.

    ``samples=1`` (default): a single call, retried once on a malformed response.
    ``samples>1``: each of the ``samples`` calls is independently retried once;
    failed samples are dropped (a `RuntimeError` only if *all* of them fail); the
    survivors are combined field-by-field (median/majority vote, see `_aggregate`).
    Even at `temperature=0` (see `llm_backends.py`), a single LLM call has real
    sample-to-sample noise on the harder-to-calibrate numeric fields (grip/crush
    force especially) -- aggregating several measurably reduces it.
    """
    results = [_extract_with_retry(prompt, backend) for _ in range(max(samples, 1))]
    good = [r for r in results if r is not None]
    if not good:
        raise RuntimeError(f"{name}: all {samples} sample(s) were malformed, even after retrying each once.")
    if len(good) < samples:
        LOGGER.warning("%s: only %d/%d samples were well-formed; aggregating those.", name, len(good), samples)

    fields = good[0]
    if len(good) > 1:
        fields = _aggregate(good)
        _log_spread(name, good, fields)
    return _build(name, fields)
