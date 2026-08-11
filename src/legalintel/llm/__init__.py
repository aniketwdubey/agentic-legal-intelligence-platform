"""LLM provider factory."""

from __future__ import annotations

from legalintel.config import Settings
from legalintel.llm.base import LLMClient, LLMError, LLMTransientError, structured
from legalintel.llm.mock import MockLLMClient

__all__ = ["LLMClient", "LLMError", "LLMTransientError", "structured", "build_llm_client"]


def build_llm_client(settings: Settings) -> LLMClient:
    """Return the configured LLM client. ``mock`` needs no AWS deps."""
    if settings.llm_provider == "bedrock":
        from legalintel.llm.bedrock import BedrockClient

        return BedrockClient(settings)
    return MockLLMClient()
