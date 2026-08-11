"""Application configuration.

All settings come from the environment (12-factor) via ``pydantic-settings``.
Nothing here reads secrets from source. A single frozen ``settings`` instance is
imported across the app; tests build their own ``Settings(...)`` to override.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["mock", "bedrock"]
Embedder = Literal["hashing", "bedrock"]
LogFormat = Literal["json", "console"]


class Settings(BaseSettings):
    """Typed, validated application settings loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_prefix="LEGALINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = "mock"
    aws_region: str = "us-east-1"
    # Haiku 4.5 is the cheapest current Claude on Bedrock ($1/$5 per M tokens) and
    # is plenty for this grounded-drafting task. Current Claude models are NOT
    # on-demand invokable by their bare id — you must use a cross-region inference
    # profile id (us./global. prefix). Run scripts/check_bedrock.py to see the ids
    # available to your account; step up to us.anthropic.claude-sonnet-5 etc. for
    # more headroom. Requires one-time Bedrock model-access approval in the console.
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"

    # --- Embedder ----------------------------------------------------------
    embedder: Embedder = "hashing"

    # --- Corpus ------------------------------------------------------------
    corpus_dir: str = "data/corpus"
    corpus_s3_bucket: str = ""

    # --- Retrieval / grounding --------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    hybrid_dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding_threshold: float = Field(default=0.18, ge=0.0, le=1.0)

    # --- Reliability guardrails -------------------------------------------
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Observability -----------------------------------------------------
    log_format: LogFormat = "console"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
