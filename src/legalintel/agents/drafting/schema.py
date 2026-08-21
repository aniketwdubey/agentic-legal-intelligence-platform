"""Drafting output schema — the ungrounded-until-validated draft answer."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
