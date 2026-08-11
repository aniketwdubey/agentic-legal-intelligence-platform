"""Retrieval package: corpus loading + hybrid (BM25 + dense) search."""

from legalintel.retrieval.corpus import load_corpus
from legalintel.retrieval.hybrid import HybridRetriever

__all__ = ["load_corpus", "HybridRetriever"]
