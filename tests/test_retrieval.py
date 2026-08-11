"""Retrieval logic: BM25, hybrid fusion, and query routing."""

from __future__ import annotations

from legalintel.retrieval.bm25 import BM25
from legalintel.retrieval.hybrid import HybridRetriever
from legalintel.retrieval.text import token_overlap, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    toks = tokenize("The fair use of a copyrighted work!")
    assert "the" not in toks and "of" not in toks
    assert "fair" in toks and "copyrighted" in toks


def test_token_overlap_bounds():
    assert token_overlap("", "anything") == 0.0
    assert token_overlap("fair use", "the fair use of a work") == 1.0
    assert 0.0 < token_overlap("fair market value", "the fair use") < 1.0


def test_bm25_ranks_matching_doc_first():
    docs = [
        "confidentiality obligation receiving party strict confidence",
        "fair use copyrighted work criticism comment teaching research",
    ]
    bm25 = BM25(docs)
    scores = bm25.scores("fair use of a copyrighted work")
    assert scores[1] > scores[0]


def test_hybrid_search_returns_relevant_authority(retriever: HybridRetriever):
    results = retriever.search("fair use of a copyrighted work")
    assert results, "expected at least one hit"
    assert results[0].authority.id == "fix-statute-a"
    assert results[0].score > 0


def test_hybrid_search_many_dedupes_by_authority(retriever: HybridRetriever):
    results = retriever.search_many(["fair use copyrighted work", "fair use factors market effect"])
    ids = [r.authority.id for r in results]
    assert len(ids) == len(set(ids)), "authorities must be de-duplicated across queries"


def test_empty_corpus_returns_no_hits(settings):
    empty = HybridRetriever([], settings)
    assert empty.search("anything") == []
