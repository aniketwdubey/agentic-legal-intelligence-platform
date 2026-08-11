"""Eval metrics: retrieval recall@k and citation-quality scores.

Kept as pure functions so they are unit-testable and reusable by a future CI
gate. The headline metric is ``hallucinated_citation_rate`` — the fraction of
emitted citations not present in the retrieved set — which the platform is
designed to drive to ~0.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(expected: list[str], retrieved_ids: list[str], k: int) -> float:
    """Fraction of expected authorities present in the top-k retrieved ids."""
    if not expected:
        return 1.0
    topk = set(retrieved_ids[:k])
    return sum(1 for e in expected if e in topk) / len(expected)


def citation_precision(emitted: list[str], expected: list[str]) -> float:
    """Of the citations emitted, the fraction that are expected/correct."""
    if not emitted:
        return 1.0  # no wrong citations emitted
    exp = set(expected)
    return sum(1 for c in emitted if c in exp) / len(emitted)


def hallucination_rate(emitted: list[str], retrieved_ids: list[str]) -> float:
    """Fraction of emitted citations NOT present in the retrieved set."""
    if not emitted:
        return 0.0
    got = set(retrieved_ids)
    return sum(1 for c in emitted if c not in got) / len(emitted)


@dataclass
class EvalSummary:
    n: int
    mean_recall_at_k: float
    mean_citation_precision: float
    hallucinated_citation_rate: float
    correct_abstention_rate: float
    answered: int
    abstained: int
    escalated: int

    def as_table(self) -> str:
        rows = [
            ("items", self.n),
            ("recall@k", f"{self.mean_recall_at_k:.3f}"),
            ("citation_precision", f"{self.mean_citation_precision:.3f}"),
            ("hallucinated_citation_rate", f"{self.hallucinated_citation_rate:.3f}"),
            ("correct_abstention_rate", f"{self.correct_abstention_rate:.3f}"),
            ("answered/abstained/escalated", f"{self.answered}/{self.abstained}/{self.escalated}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k.ljust(width)} : {v}" for k, v in rows)
