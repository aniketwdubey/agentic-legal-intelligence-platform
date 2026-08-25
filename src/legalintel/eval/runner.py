"""Eval runner (script).

Usage:
    python -m legalintel.eval.runner eval/golden_set.jsonl [--fail-under-recall 0.6]
                                                           [--max-hallucination 0.0]

Runs the full pipeline over the golden set and reports retrieval recall@k,
citation precision, hallucinated-citation rate, and correct-abstention rate.
Wire the optional thresholds into CI later to gate regressions; by default it
only reports (exit 0).
"""

from __future__ import annotations

import argparse
import sys

from legalintel.agents import build_supervisor
from legalintel.config import get_settings
from legalintel.eval.golden import load_golden
from legalintel.eval.metrics import (
    EvalSummary,
    citation_precision,
    hallucination_rate,
    recall_at_k,
)
from legalintel.retrieval import HybridRetriever, load_corpus
from legalintel.schemas import QueryRequest, Status


def run_eval(golden_path: str) -> EvalSummary:
    settings = get_settings()
    corpus = load_corpus(settings)
    retriever = HybridRetriever(corpus, settings)
    supervisor = build_supervisor(settings)
    items = load_golden(golden_path)

    recalls, precisions, halluc = [], [], []
    correct_abstentions = 0
    n_abstain_expected = 0
    answered = abstained = escalated = 0

    for item in items:
        retrieved_ids = [r.authority.id for r in retriever.search_many([item.question])]
        recalls.append(
            recall_at_k(item.expected_authorities, retrieved_ids, settings.retrieval_top_k)
        )

        resp = supervisor.run(QueryRequest(question=item.question, jurisdiction=item.jurisdiction))
        emitted = [c.authority_id for c in resp.citations]
        precisions.append(citation_precision(emitted, item.expected_authorities))
        halluc.append(hallucination_rate(emitted, retrieved_ids))

        if resp.status is Status.ANSWERED:
            answered += 1
        elif resp.status is Status.ABSTAINED:
            abstained += 1
        else:
            escalated += 1

        if item.expect_abstention:
            n_abstain_expected += 1
            if resp.status is Status.ABSTAINED:
                correct_abstentions += 1

    n = len(items) or 1
    return EvalSummary(
        n=len(items),
        mean_recall_at_k=sum(recalls) / n,
        mean_citation_precision=sum(precisions) / n,
        hallucinated_citation_rate=sum(halluc) / n,
        correct_abstention_rate=(correct_abstentions / n_abstain_expected)
        if n_abstain_expected
        else 1.0,
        answered=answered,
        abstained=abstained,
        escalated=escalated,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the legal-intel eval harness.")
    parser.add_argument("golden", help="Path to golden-set JSONL.")
    parser.add_argument("--fail-under-recall", type=float, default=None)
    parser.add_argument("--max-hallucination", type=float, default=None)
    args = parser.parse_args(argv)

    summary = run_eval(args.golden)
    print("\n=== Legal Intelligence — Eval Summary ===")
    print(summary.as_table())

    failed = False
    if args.fail_under_recall is not None and summary.mean_recall_at_k < args.fail_under_recall:
        print(f"\nFAIL: recall@k {summary.mean_recall_at_k:.3f} < {args.fail_under_recall}")
        failed = True
    if (
        args.max_hallucination is not None
        and summary.hallucinated_citation_rate > args.max_hallucination
    ):
        print(
            f"\nFAIL: hallucination rate {summary.hallucinated_citation_rate:.3f} "
            f"> {args.max_hallucination}"
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
