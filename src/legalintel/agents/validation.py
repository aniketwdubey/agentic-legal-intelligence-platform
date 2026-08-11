"""Validation agent — the citation-verification core of the platform.

This is deliberately **rule-based, not another LLM call**. Verifying a citation
is a factual check against the retrieved set, and we do not want a second
hallucination-prone model deciding whether the first one hallucinated. For each
claim it checks, independently:

1. **citation_exists** — is ``authority_id`` actually in the retrieved set?
   (A "no" is a hallucinated citation — the famous failure mode this platform
   exists to prevent.)
2. **supported** — does the claim's quote/text overlap the cited authority's
   text above ``grounding_threshold``? (Catches a real citation attached to an
   unsupported proposition.)

The supervisor uses this report to decide answer vs. abstain vs. escalate.
"""

from __future__ import annotations

from dataclasses import dataclass

from legalintel.agents.base import Agent
from legalintel.config import Settings
from legalintel.retrieval.text import token_overlap
from legalintel.schemas import (
    ClaimVerdict,
    DraftAnswer,
    RetrievedAuthority,
    ValidationReport,
)


@dataclass
class ValidationInput:
    draft: DraftAnswer
    retrieved: list[RetrievedAuthority]


class ValidationAgent(Agent[ValidationInput, ValidationReport]):
    name = "validation"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._threshold = settings.grounding_threshold

    def run(self, inp: ValidationInput) -> ValidationReport:
        by_id = {r.authority.id: r.authority for r in inp.retrieved}
        verdicts: list[ClaimVerdict] = []

        for claim in inp.draft.claims:
            authority = by_id.get(claim.authority_id)

            if authority is None:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim,
                        citation_exists=False,
                        supported=False,
                        support_score=0.0,
                        reason=f"authority_id '{claim.authority_id}' not in retrieved set "
                        "(hallucinated citation)",
                    )
                )
                self.log.warning(
                    "validation.hallucinated_citation", authority_id=claim.authority_id
                )
                continue

            # Two independent checks against the cited source text:
            #   * quote_authentic — the quote is actually a span of the source
            #     (guards a *fabricated* quote pinned to a real citation);
            #   * relevance — the claim's proposition is covered by the source
            #     above threshold (guards a real citation that does not in fact
            #     support the proposition).
            source_text = authority.text
            quote_cov = token_overlap(claim.quote, source_text) if claim.quote else 1.0
            quote_authentic = quote_cov >= 0.5
            relevance = token_overlap(claim.text, source_text)
            support_score = relevance
            supported = quote_authentic and relevance >= self._threshold

            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    citation_exists=True,
                    supported=supported,
                    support_score=round(min(support_score, 1.0), 4),
                    reason=(
                        "grounded"
                        if supported
                        else "quote not found in cited authority (fabricated quote)"
                        if not quote_authentic
                        else "cited authority does not support the proposition"
                    ),
                )
            )

        report = ValidationReport(verdicts=verdicts)
        self.log.info(
            "validation.report",
            total=len(verdicts),
            hallucinated=len(report.hallucinated_citations),
            unsupported=len(report.unsupported_claims),
            all_verified=report.all_verified,
        )
        return report
