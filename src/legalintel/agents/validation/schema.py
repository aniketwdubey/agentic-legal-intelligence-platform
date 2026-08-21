"""Validation output schema — per-claim verdicts and their rollups."""

from __future__ import annotations

from pydantic import BaseModel, Field

from legalintel.agents.drafting.schema import Claim


class ClaimVerdict(BaseModel):
    claim: Claim
    citation_exists: bool = Field(description="authority_id is in the retrieved set.")
    supported: bool = Field(description="Quote/claim is grounded above threshold.")
    support_score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ValidationReport(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)

    @property
    def hallucinated_citations(self) -> list[Claim]:
        return [v.claim for v in self.verdicts if not v.citation_exists]

    @property
    def unsupported_claims(self) -> list[Claim]:
        return [v.claim for v in self.verdicts if v.citation_exists and not v.supported]

    @property
    def all_verified(self) -> bool:
        return bool(self.verdicts) and all(
            v.citation_exists and v.supported for v in self.verdicts
        )
