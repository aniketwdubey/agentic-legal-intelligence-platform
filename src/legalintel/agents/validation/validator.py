"""Citation-verification — the core guardrail of the platform.

Deliberately **rule-based, not another LLM call**. Verifying a citation is a
factual check against the retrieved set, and we do not want a second
hallucination-prone model deciding whether the first one hallucinated. For each
claim it checks, independently:

1. **citation_exists** — is ``authority_id`` actually in the retrieved set?
   (A "no" is a hallucinated citation — the failure mode this platform prevents.)
2. **supported** — is the quote a real span of the cited authority, and does the
   proposition overlap that authority above ``threshold``?

The supervisor uses this report to decide answer vs. abstain vs. escalate.
"""

from __future__ import annotations

import structlog

from legalintel.agents.drafting.schema import DraftAnswer
from legalintel.agents.validation.schema import ClaimVerdict, ValidationReport
from legalintel.retrieval.text import token_overlap
from legalintel.schemas import RetrievedAuthority

log = structlog.get_logger("agent.validation")


def validate(
    draft: DraftAnswer, retrieved: list[RetrievedAuthority], *, threshold: float
) -> ValidationReport:
    """Verify every claim in ``draft`` against the ``retrieved`` authority set."""
    by_id = {r.authority.id: r.authority for r in retrieved}
    verdicts: list[ClaimVerdict] = []

    for claim in draft.claims:
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
            log.warning("validation.hallucinated_citation", authority_id=claim.authority_id)
            continue

        # Two independent checks against the cited source text:
        #   * quote_authentic — the quote is actually a span of the source
        #     (guards a *fabricated* quote pinned to a real citation);
        #   * relevance — the claim's proposition is covered by the source above
        #     threshold (guards a real citation that does not support the proposition).
        source_text = authority.text
        quote_cov = token_overlap(claim.quote, source_text) if claim.quote else 1.0
        quote_authentic = quote_cov >= 0.5
        relevance = token_overlap(claim.text, source_text)
        support_score = relevance
        supported = quote_authentic and relevance >= threshold

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
    log.info(
        "validation.report",
        total=len(verdicts),
        hallucinated=len(report.hallucinated_citations),
        unsupported=len(report.unsupported_claims),
        all_verified=report.all_verified,
    )
    return report
