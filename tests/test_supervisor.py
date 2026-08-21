"""End-to-end supervisor behaviour with the offline StubModel and fakes."""

from __future__ import annotations

from legalintel.agents._stub_model import StubModel
from legalintel.agents.drafting import Claim, DraftAnswer, DraftingAgent
from legalintel.agents.planner import PlannerAgent
from legalintel.agents.retrieval import make_retrieve_tool
from legalintel.agents.supervisor import Supervisor, build_supervisor
from legalintel.schemas import QueryRequest, Status


def _supervisor(model, retriever, settings) -> Supervisor:
    return Supervisor(
        planner=PlannerAgent(model),
        retrieve=make_retrieve_tool(retriever),
        drafting=DraftingAgent(model),
        grounding_threshold=settings.grounding_threshold,
    )


def test_grounded_query_is_answered_with_verified_citations(retriever, settings):
    sup = _supervisor(StubModel(), retriever, settings)
    resp = sup.run(
        QueryRequest(question="What is the standard for fair use of a copyrighted work?")
    )
    assert resp.status in (Status.ANSWERED, Status.ESCALATED)
    assert resp.citations, "a grounded query must return citations"
    # Every emitted citation references a real retrieved authority (never hallucinated).
    assert all(c.authority_id == "fix-statute-a" for c in resp.citations)
    assert resp.confidence > 0
    assert resp.steps_run[:4] == ["planner", "retrieval", "drafting", "validation"]


def test_out_of_corpus_question_abstains(retriever, settings):
    sup = _supervisor(StubModel(), retriever, settings)
    resp = sup.run(
        QueryRequest(question="What is the maximum blood alcohol limit for pilots in Japan?")
    )
    assert resp.status is Status.ABSTAINED
    assert not resp.citations
    assert resp.abstention_reason


class _HallucinatingModel(StubModel):
    """Drafter that invents a citation to an authority not in the retrieved set."""

    def _draft(self, text: str) -> DraftAnswer:
        return DraftAnswer(
            answer="Fabricated.",
            claims=[
                Claim(
                    text="A made-up proposition about fair use.",
                    authority_id="totally-invented-case-2099",
                    quote="nonexistent quote",
                )
            ],
        )


def test_hallucinated_citation_never_reaches_user(retriever, settings):
    sup = _supervisor(_HallucinatingModel(), retriever, settings)
    resp = sup.run(QueryRequest(question="Explain fair use of a copyrighted work."))
    # The invented citation is stripped; with no verified claim left, we abstain.
    assert resp.status is Status.ABSTAINED
    assert all(c.authority_id != "totally-invented-case-2099" for c in resp.citations)


class _FailingModel(StubModel):
    """Model that fails during planning (simulates a Bedrock outage)."""

    def _plan(self, text: str):
        raise RuntimeError("bedrock down")


def test_model_failure_escalates(retriever, settings):
    sup = _supervisor(_FailingModel(), retriever, settings)
    resp = sup.run(QueryRequest(question="Anything about fair use."))
    assert resp.status is Status.ESCALATED
    assert resp.abstention_reason


def test_build_supervisor_from_settings(settings):
    sup = build_supervisor(settings)
    resp = sup.run(QueryRequest(question="confidentiality obligation of the receiving party"))
    assert resp.status in (Status.ANSWERED, Status.ESCALATED, Status.ABSTAINED)
    assert resp.trace_id
