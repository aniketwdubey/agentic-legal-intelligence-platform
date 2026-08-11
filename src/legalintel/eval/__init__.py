"""Evaluation harness: golden-set format, metrics, and runner."""

from legalintel.eval.golden import GoldenItem, load_golden
from legalintel.eval.metrics import (
    EvalSummary,
    citation_precision,
    hallucination_rate,
    recall_at_k,
)

__all__ = [
    "GoldenItem",
    "load_golden",
    "EvalSummary",
    "citation_precision",
    "hallucination_rate",
    "recall_at_k",
]
