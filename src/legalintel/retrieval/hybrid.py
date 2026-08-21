"""Hybrid retriever: BM25 (lexical) + dense (semantic), fused per query.

Each leg is min-max normalised to [0, 1] over the candidate set, then combined
as ``dense_weight * dense + (1 - dense_weight) * bm25``. This makes the two
incomparable score scales comparable and lets one config knob trade lexical vs
semantic recall. Indexing embeds the corpus once at construction.
"""

from __future__ import annotations

import numpy as np
import structlog

from legalintel.config import Settings
from legalintel.retrieval.bm25 import BM25
from legalintel.retrieval.embeddings import Embedder, build_embedder
from legalintel.schemas import Authority, RetrievedAuthority

log = structlog.get_logger("retrieval")


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


class HybridRetriever:
    def __init__(
        self, corpus: list[Authority], settings: Settings, embedder: Embedder | None = None
    ):
        self._corpus = corpus
        self._settings = settings
        self._embedder = embedder or build_embedder(settings)
        # Index over "citation + title + text" so citation tokens are searchable.
        self._docs = [f"{a.citation}\n{a.title}\n{a.text}" for a in corpus]
        self._bm25 = BM25(self._docs)
        self._doc_vecs = (
            self._embedder.embed(self._docs) if corpus else np.zeros((0, 1), dtype=np.float32)
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedAuthority]:
        if not self._corpus:
            return []
        k = top_k or self._settings.retrieval_top_k
        w = self._settings.hybrid_dense_weight

        bm25 = _minmax(np.array(self._bm25.scores(query), dtype=np.float32))
        q_vec = self._embedder.embed([query])[0]
        dense_raw = self._doc_vecs @ q_vec  # cosine (both L2-normalised)
        dense = _minmax(dense_raw)

        fused = w * dense + (1.0 - w) * bm25
        order = np.argsort(-fused)[:k]

        results = [
            RetrievedAuthority(
                authority=self._corpus[i],
                score=round(float(fused[i]), 6),
                bm25_score=round(float(bm25[i]), 6),
                dense_score=round(float(dense_raw[i]), 6),
            )
            for i in order
            if fused[i] > 0
        ]
        log.info("retrieval.search", query=query, hits=len(results), top_k=k)
        return results

    def search_many(self, queries: list[str], top_k: int | None = None) -> list[RetrievedAuthority]:
        """Union results across queries, keeping the best score per authority."""
        best: dict[str, RetrievedAuthority] = {}
        for q in queries:
            for r in self.search(q, top_k=top_k):
                cur = best.get(r.authority.id)
                if cur is None or r.score > cur.score:
                    best[r.authority.id] = r
        return sorted(best.values(), key=lambda r: -r.score)[
            : (top_k or self._settings.retrieval_top_k)
        ]
