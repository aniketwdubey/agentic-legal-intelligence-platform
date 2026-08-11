"""LLM client protocol and the guarded structured-output helper.

Every model call in the platform goes through ``structured()``, which enforces
the three guardrails the domain demands:

* **timeout**  — no call blocks the request past ``llm_timeout_seconds``;
* **retry w/ backoff** — transient errors AND schema-invalid outputs are retried
  with exponential backoff (tenacity);
* **schema validation** — the raw completion is parsed into the caller's pydantic
  model; if it doesn't validate, that's a retryable failure, not a silent pass.

Swapping Bedrock for a local/mock client is a one-line provider change; the
guardrails live here, not in the provider.
"""

from __future__ import annotations

import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from legalintel.config import Settings
from legalintel.logging import get_logger

log = get_logger("llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Non-retryable LLM failure (auth, bad request)."""


class LLMTransientError(RuntimeError):
    """Retryable LLM failure (throttling, timeout, malformed output)."""


class LLMClient(Protocol):
    """Minimal text-in/JSON-text-out contract every provider implements."""

    def complete_json(self, *, system: str, prompt: str, timeout: float) -> str:
        """Return a raw JSON string completion. Raise LLM*Error on failure."""
        ...


def _extract_json(raw: str) -> str:
    """Best-effort extraction of the first JSON object from a completion."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMTransientError("no JSON object found in completion")
    return raw[start : end + 1]


def structured(
    client: LLMClient,
    *,
    settings: Settings,
    system: str,
    prompt: str,
    schema: type[T],
) -> T:
    """Call the LLM and return a validated instance of ``schema``.

    Retries on transient provider errors *and* on schema-invalid completions,
    with exponential backoff, up to ``settings.llm_max_retries`` attempts.
    """

    @retry(
        retry=retry_if_exception_type(LLMTransientError),
        stop=stop_after_attempt(settings.llm_max_retries + 1),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def _call() -> T:
        raw = client.complete_json(
            system=system, prompt=prompt, timeout=settings.llm_timeout_seconds
        )
        try:
            return schema.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError) as exc:
            log.warning("llm.schema_invalid", schema=schema.__name__, error=str(exc))
            # Re-raise as transient so tenacity retries with a fresh completion.
            raise LLMTransientError(f"schema validation failed: {exc}") from exc

    return _call()
