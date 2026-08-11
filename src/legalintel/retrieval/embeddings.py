"""Dense embeddings — the semantic leg of hybrid retrieval.

Two backends behind one ``Embedder`` protocol:

* ``HashingEmbedder`` — a deterministic hashed bag-of-bigrams projected into a
  fixed vector space. Zero cost, zero network, so it is the default for local
  runs, tests, and CI. It captures lexical/semantic overlap well enough to make
  the hybrid fusion meaningful without a model.
* ``BedrockEmbedder`` — Amazon Titan text embeddings (pay-per-call), for the
  quality bar in deployed environments.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from legalintel.config import Settings
from legalintel.retrieval.text import tokenize

_DIM = 256


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalised
        ...


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.asarray(mat / np.clip(norms, 1e-9, None), dtype=np.float32)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        # Unigrams + bigrams hashed into buckets with a sign hash (signed hashing
        # trick) to reduce collisions cancelling out.
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        for g in grams:
            h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
            bucket = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[bucket] += sign
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vec(t) for t in texts]))


class BedrockEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        import boto3

        self._settings = settings
        self._model_id = settings.bedrock_embed_model_id
        self._client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def embed(self, texts: list[str]) -> np.ndarray:
        import json

        vecs = []
        for t in texts:
            resp = self._client.invoke_model(
                modelId=self._model_id, body=json.dumps({"inputText": t})
            )
            vecs.append(json.loads(resp["body"].read())["embedding"])
        return _normalize(np.array(vecs, dtype=np.float32))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "bedrock":
        return BedrockEmbedder(settings)
    return HashingEmbedder()
