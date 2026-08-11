"""Pydantic schemas — the typed contracts at every agent boundary.

Each agent declares an input and output schema. The supervisor validates
outputs against these schemas on every hop, so a malformed model response is
caught at the boundary that produced it (debuggability) rather than surfacing
as a corrupt final answer.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Corpus / retrieval primitives
# ---------------------------------------------------------------------------
class AuthorityType(str, Enum):
    STATUTE = "statute"
    CASE = "case"
    REGULATION = "regulation"
    CONTRACT_CLAUSE = "contract_clause"


class Authority(BaseModel):
    """A single unit of legal authority in the corpus (statute, case, clause)."""

    id: str = Field(description="Stable corpus id, e.g. 'usc-17-107'.")
    citation: str = Field(description="Human citation, e.g. '17 U.S.C. § 107'.")
    type: AuthorityType
    jurisdiction: str = Field(default="US", description="e.g. 'US', 'US-CA', 'UK'.")
    title: str
    text: str
    source_url: str | None = None


class RetrievedAuthority(BaseModel):
    """An authority returned by retrieval with its fused relevance score."""

    authority: Authority
    score: float = Field(ge=0.0, description="Fused hybrid relevance score.")
    bm25_score: float = 0.0
    dense_score: float = 0.0


# ---------------------------------------------------------------------------
# Task intent (planner output)
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    RESEARCH = "research"
    DRAFTING = "drafting"
    DOC_REVIEW = "doc_review"


class PlanStep(str, Enum):
    RETRIEVE = "retrieve"
    DRAFT = "draft"
    VALIDATE = "validate"


class Plan(BaseModel):
    """Planner output: the interpreted intent and the ordered steps to run."""

    task_type: TaskType
    jurisdiction: str = "US"
    search_queries: list[str] = Field(
        min_length=1, description="Queries the retrieval agent should issue."
    )
    steps: list[PlanStep] = Field(min_length=1)
    rationale: str = Field(description="Why the planner chose these steps.")


# ---------------------------------------------------------------------------
# Drafting agent output
# ---------------------------------------------------------------------------
class Claim(BaseModel):
    """A single legal proposition the drafting agent asserts, with its citation.

    ``authority_id`` MUST reference an id present in the retrieved set; the
    validation agent enforces this and computes support.
    """

    text: str = Field(description="The legal proposition being asserted.")
    authority_id: str = Field(description="Corpus id of the supporting authority.")
    quote: str = Field(default="", description="Verbatim span from the authority text.")


class DraftAnswer(BaseModel):
    """Ungrounded-until-validated output of the drafting agent."""

    answer: str = Field(description="Prose answer / draft assembled from claims.")
    claims: list[Claim] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation agent output
# ---------------------------------------------------------------------------
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
        return bool(self.verdicts) and all(v.citation_exists and v.supported for v in self.verdicts)


# ---------------------------------------------------------------------------
# Public API request / response
# ---------------------------------------------------------------------------
class Status(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    jurisdiction: str | None = None


class CitationOut(BaseModel):
    citation: str
    authority_id: str
    quote: str = ""
    source_url: str | None = None


class QueryResponse(BaseModel):
    """The only thing a user ever sees. An answer here is citation-verified."""

    status: Status
    trace_id: str
    task_type: TaskType | None = None
    answer: str = ""
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstention_reason: str | None = None
    steps_run: list[str] = Field(default_factory=list)
