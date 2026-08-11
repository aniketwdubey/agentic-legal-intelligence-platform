"""Shared text utilities: tokenisation and token-overlap scoring.

Kept dependency-free and deterministic so both retrieval and the rule-based
validation agent share exactly one notion of "tokens".
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9§]+")

# Legal/English stopwords trimmed to keep BM25 focused on content terms.
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "but",
        "is",
        "are",
        "be",
        "as",
        "by",
        "with",
        "at",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "shall",
        "may",
        "must",
        "not",
        "no",
        "any",
        "all",
        "such",
        "under",
        "over",
        "per",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


def token_overlap(claim: str, source: str) -> float:
    """Jaccard-style overlap of claim tokens covered by the source in [0, 1]."""
    ct = set(tokenize(claim))
    if not ct:
        return 0.0
    st = set(tokenize(source))
    return len(ct & st) / len(ct)
