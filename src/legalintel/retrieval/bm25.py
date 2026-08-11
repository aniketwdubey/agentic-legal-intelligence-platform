"""Okapi BM25 — the lexical (sparse) leg of hybrid retrieval.

Pure-Python, no heavy deps, so it runs in CI and on Lambda without a search
service. Good at exact statute/citation term matches where dense embeddings
under-weight rare tokens.
"""

from __future__ import annotations

import math
from collections import Counter

from legalintel.retrieval.text import tokenize


class BM25:
    def __init__(self, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tok_docs = [tokenize(d) for d in docs]
        self._doc_len = [len(d) for d in self._tok_docs]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._freqs = [Counter(d) for d in self._tok_docs]
        self._idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._tok_docs)
        df: Counter[str] = Counter()
        for doc in self._tok_docs:
            df.update(set(doc))
        # Robertson-Sparck-Jones idf with +1 to keep weights non-negative.
        return {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        out: list[float] = []
        for i, freq in enumerate(self._freqs):
            dl = self._doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            s = 0.0
            for t in q_tokens:
                if t not in freq:
                    continue
                tf = freq[t]
                s += self._idf.get(t, 0.0) * tf * (self.k1 + 1) / (tf + denom_norm)
            out.append(s)
        return out
