"""Agent-layer exceptions."""

from __future__ import annotations


class GuardrailBlocked(RuntimeError):
    """A Bedrock Guardrail intervened on a model call (e.g. prompt injection / PII).

    Raised by an LLM agent when the model response stops with
    ``guardrail_intervened``; the supervisor turns it into a clean abstention
    rather than a generic pipeline failure.
    """
