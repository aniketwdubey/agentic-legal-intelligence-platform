"""API smoke tests via FastAPI's TestClient (offline mock provider)."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Point the app at the offline mock provider + fixture corpus before import.
FIXTURE_CORPUS = os.path.join(os.path.dirname(__file__), "fixtures", "corpus")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LEGALINTEL_LLM_PROVIDER", "mock")
    monkeypatch.setenv("LEGALINTEL_EMBEDDER", "hashing")
    monkeypatch.setenv("LEGALINTEL_CORPUS_DIR", FIXTURE_CORPUS)
    # Fresh settings + app so env overrides take effect.
    from legalintel.config import get_settings

    get_settings.cache_clear()
    from legalintel.api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_verified_response(client):
    r = client.post("/v1/query", json={"question": "What is fair use of a copyrighted work?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"answered", "escalated", "abstained"}
    assert body["trace_id"]
    for c in body["citations"]:
        assert c["authority_id"] == "fix-statute-a"


def test_query_rejects_empty_question(client):
    r = client.post("/v1/query", json={"question": ""})
    assert r.status_code == 422  # schema validation at the boundary
