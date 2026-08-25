"""Shared pytest fixtures. Everything here is offline (mock LLM, local corpus)."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalintel.config import Settings
from legalintel.retrieval import HybridRetriever, load_corpus
from legalintel.schemas import Authority

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture
def settings() -> Settings:
    """Settings pinned to the offline mock provider and the test fixture corpus."""
    return Settings(
        llm_provider="mock",
        embedder="hashing",
        corpus_dir=str(FIXTURE_CORPUS),
        retrieval_top_k=5,
        grounding_threshold=0.18,
    )


@pytest.fixture
def corpus(settings: Settings) -> list[Authority]:
    return load_corpus(settings)


@pytest.fixture
def retriever(corpus: list[Authority], settings: Settings) -> HybridRetriever:
    return HybridRetriever(corpus, settings)
