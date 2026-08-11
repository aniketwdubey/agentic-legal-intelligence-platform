"""End-to-end supervisor behaviour with the offline mock provider and fakes."""

from __future__ import annotations

import json

from legalintel.agents.drafting import DraftingAgent
from legalintel.agents.planner import PlannerAgent
from legalintel.agents.retrieval import RetrievalAgent
from legalintel.agents.supervisor import Supervisor, build_supervisor
from legalintel.agents.validation import ValidationAgent
from legalintel.llm import LLMError
from legalintel.llm.mock import MockLLMClient
from legalintel.llm.prompts import ROLE_DRAFTER, ROLE_PLANNER
from legalintel.schemas import QueryRequest, Status


def _supervisor(client, retriever, settings) -> Supervisor:
    return Supervisor(
        planner=PlannerAgent(client, settings),
        retrieval=RetrievalAgent(retriever),
        drafting=DraftingAgent(client, settings),
        validation=ValidationAgent(settings),
    )


def test_grounded_query_is_answered_with_verified_citations(retriever, settings):
    sup = _supervisor(MockLLMClient(), retriever, settings)
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
    sup = _supervisor(MockLLMClient(), retriever, settings)
    resp = sup.run(
        QueryRequest(question="What is the maximum blood alcohol limit for pilots in Japan?")
    )
    assert resp.status is Status.ABSTAINED
    assert not resp.citations
    assert resp.abstention_reason


class _HallucinatingClient(MockLLMClient):
    """Drafter that invents a citation to an authority not in the retrieved set."""

    def complete_json(self, *, system, prompt, timeout):
        if ROLE_DRAFTER in system:
            return json.dumps(
                {
                    "answer": "Fabricated.",
                    "claims": [
                        {
                            "text": "A made-up proposition about fair use.",
                            "authority_id": "totally-invented-case-2099",
                            "quote": "nonexistent quote",
                        }
                    ],
                }
            )
        return super().complete_json(system=system, prompt=prompt, timeout=timeout)


def test_hallucinated_citation_never_reaches_user(retriever, settings):
    sup = _supervisor(_HallucinatingClient(), retriever, settings)
    resp = sup.run(QueryRequest(question="Explain fair use of a copyrighted work."))
    # The invented citation is stripped; with no verified claim left, we abstain.
    assert resp.status is Status.ABSTAINED
    assert all(c.authority_id != "totally-invented-case-2099" for c in resp.citations)


class _FailingClient(MockLLMClient):
    def complete_json(self, *, system, prompt, timeout):
        if ROLE_PLANNER in system:
            raise LLMError("bedrock down")
        return super().complete_json(system=system, prompt=prompt, timeout=timeout)


def test_non_retryable_llm_error_escalates(retriever, settings):
    sup = _supervisor(_FailingClient(), retriever, settings)
    resp = sup.run(QueryRequest(question="Anything about fair use."))
    assert resp.status is Status.ESCALATED
    assert resp.abstention_reason


def test_build_supervisor_from_settings(settings):
    sup = build_supervisor(settings)
    resp = sup.run(QueryRequest(question="confidentiality obligation of the receiving party"))
    assert resp.status in (Status.ANSWERED, Status.ESCALATED, Status.ABSTAINED)
    assert resp.trace_id
