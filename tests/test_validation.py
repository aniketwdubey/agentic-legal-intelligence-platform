"""Citation-verification agent — the platform's core guardrail."""

from __future__ import annotations

import pytest

from legalintel.agents.validation import ValidationAgent, ValidationInput
from legalintel.schemas import Authority, AuthorityType, Claim, DraftAnswer, RetrievedAuthority


@pytest.fixture
def authority() -> Authority:
    return Authority(
        id="a1",
        citation="Test Stat. § 1",
        type=AuthorityType.STATUTE,
        title="Fair use",
        text="The fair use of a copyrighted work for criticism, comment, teaching, or research "
        "is not an infringement of copyright.",
    )


@pytest.fixture
def retrieved(authority: Authority) -> list[RetrievedAuthority]:
    return [RetrievedAuthority(authority=authority, score=0.9)]


def _validate(claims, retrieved, settings):
    agent = ValidationAgent(settings)
    return agent.run(
        ValidationInput(draft=DraftAnswer(answer="x", claims=claims), retrieved=retrieved)
    )


def test_grounded_claim_is_verified(retrieved, settings):
    claim = Claim(
        text="Fair use of a copyrighted work for teaching is not an infringement of copyright.",
        authority_id="a1",
        quote="The fair use of a copyrighted work for criticism, comment, teaching, or research",
    )
    report = _validate([claim], retrieved, settings)
    assert report.all_verified
    assert report.verdicts[0].supported
    assert not report.hallucinated_citations


def test_hallucinated_citation_is_flagged(retrieved, settings):
    """A citation to an authority not in the retrieved set is the headline failure."""
    claim = Claim(text="Some proposition.", authority_id="does-not-exist", quote="")
    report = _validate([claim], retrieved, settings)
    assert not report.verdicts[0].citation_exists
    assert report.hallucinated_citations
    assert not report.all_verified


def test_fabricated_quote_on_real_citation_is_unsupported(retrieved, settings):
    """Real authority_id but a quote that is not actually in the source text."""
    claim = Claim(
        text="Fair use is not infringement.",
        authority_id="a1",
        quote="This exact sentence appears nowhere in the statute about zoning permits.",
    )
    report = _validate([claim], retrieved, settings)
    v = report.verdicts[0]
    assert v.citation_exists and not v.supported
    assert "fabricated quote" in v.reason


def test_irrelevant_claim_on_real_citation_is_unsupported(retrieved, settings):
    """Real citation, authentic-looking but the proposition is unrelated."""
    claim = Claim(
        text="Airline pilots must not exceed a blood alcohol concentration of 0.02 percent.",
        authority_id="a1",
        quote="",
    )
    report = _validate([claim], retrieved, settings)
    assert report.verdicts[0].citation_exists
    assert not report.verdicts[0].supported


def test_empty_report_is_not_verified(settings):
    report = _validate([], [], settings)
    assert not report.all_verified
