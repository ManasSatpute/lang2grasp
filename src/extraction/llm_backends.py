"""LLM backends for turning a text prompt into raw object-parameter JSON.

Every backend implements the same one-method interface and returns a plain
``dict`` matching :class:`objects.object_params.ObjectParams`'s fields
(``shape``, ``size``, ``density``, ``friction``, ``mass_class``, ``fragile``,
``grip_force_min_N``, ``grip_force_max_N``, ``rgba``, ``spring_Npm``,
``crush_force_N``) -- validation and clamping happen once, in
``ObjectParams.__post_init__``, not here.

SDKs are imported lazily inside each backend's ``__init__`` (same pattern as
``extraction/deligrasp/llm.py``'s ``OpenAIBackend``) so ``requirements.txt``
doesn't have to hard-depend on any of them -- ``MockBackend`` needs none and
is the default for tests, CI, and SLURM nodes with no API key or network.
"""

from __future__ import annotations

import json
import re

from extraction.param_prompts import JSON_SCHEMA, SYSTEM_PROMPT, prior_for


class ParamBackend:
    def extract(self, prompt: str) -> dict:
        """Return raw JSON fields for an object described by ``prompt``."""
        raise NotImplementedError


class MockBackend(ParamBackend):
    """Deterministic, offline stand-in. Keyword-matches ``prompt`` against a
    small table of plausible priors -- no API key, no network, fully
    reproducible. Default backend for tests and SLURM smoke runs.
    """

    def extract(self, prompt: str) -> dict:
        _, fields = prior_for(prompt)
        return fields


class AnthropicBackend(ParamBackend):
    """Claude, via ``output_config.format`` structured JSON output."""

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def extract(self, prompt: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": JSON_SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


class OpenAIBackend(ParamBackend):
    """OpenAI, via Chat Completions ``response_format`` JSON-schema mode."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def extract(self, prompt: str) -> dict:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "object_params", "schema": JSON_SCHEMA, "strict": True},
            },
        )
        return json.loads(completion.choices[0].message.content)


class GroqBackend(ParamBackend):
    """Groq (OpenAI-compatible chat API), JSON-object mode.

    Groq's JSON-schema support varies by model, so this backend uses the more
    broadly-supported ``json_object`` mode and puts the schema in the prompt
    instead of relying on server-side schema enforcement.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        from groq import Groq

        self.client = Groq()
        self.model = model

    def extract(self, prompt: str) -> dict:
        schema_hint = json.dumps(JSON_SCHEMA["properties"], indent=2)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\n\nRespond with a JSON object with exactly "
                    f"these fields:\n{schema_hint}",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content
        # Groq's json_object mode is less strict than schema-enforced modes; strip
        # any stray markdown fencing before parsing.
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        return json.loads(text)


BACKENDS: dict[str, type[ParamBackend]] = {
    "mock": MockBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "groq": GroqBackend,
}
