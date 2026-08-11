"""Unit tests for the eval metric functions."""

from __future__ import annotations

from legalintel.eval.metrics import citation_precision, hallucination_rate, recall_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "x", "b", "c"], k=4) == 1.0
    assert recall_at_k(["a", "b"], ["a", "x"], k=2) == 0.5
    assert recall_at_k([], ["a"], k=1) == 1.0  # nothing expected -> trivially recalled


def test_citation_precision():
    assert citation_precision(["a", "b"], ["a", "b", "c"]) == 1.0
    assert citation_precision(["a", "z"], ["a", "b"]) == 0.5
    assert citation_precision([], ["a"]) == 1.0  # emitted nothing wrong


def test_hallucination_rate():
    assert hallucination_rate(["a", "b"], ["a", "b", "c"]) == 0.0
    assert hallucination_rate(["a", "ghost"], ["a", "b"]) == 0.5
    assert hallucination_rate([], ["a"]) == 0.0
