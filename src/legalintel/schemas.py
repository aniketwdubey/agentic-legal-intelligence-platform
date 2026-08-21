"""Shared, cross-cutting schemas.

These are the contracts that span more than one agent: the corpus/retrieval
primitives and the public API request/response. Schemas owned by a single agent
live next to that agent (``agents/<name>/schema.py``) — e.g. ``Plan`` in the
planner, ``DraftAnswer`` in the drafter, ``ValidationReport`` in the validator.
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
# Task intent (shared: produced by the planner, surfaced in the response)
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    RESEARCH = "research"
    DRAFTING = "drafting"
    DOC_REVIEW = "doc_review"


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
