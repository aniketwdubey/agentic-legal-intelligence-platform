"""Supervisor control layer — orchestration, grounding policy, recovery.

Drives the deterministic pipeline (plan -> retrieve -> draft -> validate) and
makes the final answer/abstain/escalate decision. The core discipline: **a claim
reaches the user only if the validator verified both that its citation exists in
the retrieved set and that the cited authority supports it.** Anything less becomes
an abstention or a human-review escalation — the platform never invents law.

The individual agents run on Strands; this supervisor stays an explicit Python
workflow (not a Strands graph) precisely because the grounding gate must be
deterministic and run over the *exact* retrieved set the drafter was given.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from legalintel.agents.drafting import DraftingAgent
from legalintel.agents.errors import GuardrailBlocked
from legalintel.agents.planner import PlannerAgent
from legalintel.agents.retrieval import make_retrieve_tool
from legalintel.agents.validation import ClaimVerdict, ValidationReport, validate
from legalintel.config import Settings
from legalintel.logging import bind_run, clear_run
from legalintel.models import build_model
from legalintel.retrieval import HybridRetriever, load_corpus
from legalintel.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    RetrievedAuthority,
    Status,
    TaskType,
)

log = structlog.get_logger("supervisor")

# Below this mean support score we escalate rather than answer outright.
_LOW_CONFIDENCE = 0.35

# Exceptions that signal a programming bug rather than an operational failure. We
# let these propagate (surfacing a 500 + traceback) instead of masking them as a
# routine escalation, so defects don't hide behind a normal-looking response.
_BUG_EXCEPTIONS = (NameError, ImportError, AttributeError, TypeError, KeyError, IndexError)


@dataclass
class StepRecord:
    """One row of the execution trace — which agent, pass/fail, detail."""

    agent: str
    ok: bool
    detail: str = ""


@dataclass
class WorkflowState:
    """Shared state threaded through the run; also the trace for observability."""

    trace_id: str
    steps: list[StepRecord] = field(default_factory=list)

    def record(self, agent: str, ok: bool, detail: str = "") -> None:
        self.steps.append(StepRecord(agent=agent, ok=ok, detail=detail))


class Supervisor:
    def __init__(
        self,
        planner: PlannerAgent,
        retrieve: Any,  # a Strands @tool bound to the retriever
        drafting: DraftingAgent,
        grounding_threshold: float,
    ) -> None:
        self.planner = planner
        self.retrieve = retrieve
        self.drafting = drafting
        self._threshold = grounding_threshold

    def run(
        self, request: QueryRequest, history: list[tuple[str, str]] | None = None
    ) -> QueryResponse:
        state = WorkflowState(trace_id=uuid.uuid4().hex[:12])
        bind_run(trace_id=state.trace_id)
        try:
            return self._run(request, state, history)
        except GuardrailBlocked as exc:
            # The safety guardrail intervened (prompt injection / unsafe content).
            log.info("supervisor.guardrail_blocked", detail=str(exc))
            state.record("guardrail", ok=False, detail="blocked")
            return QueryResponse(
                status=Status.ABSTAINED,
                trace_id=state.trace_id,
                abstention_reason="request blocked by the safety guardrail "
                "(possible prompt injection or unsafe content)",
                steps_run=[s.agent for s in state.steps],
            )
        except _BUG_EXCEPTIONS:
            # A defect, not an outage — let it surface loudly rather than hide it.
            raise
        except Exception as exc:
            # Operational failure (model/network/timeout): degrade to human review —
            # the platform never returns an unverified answer or a 500 with legal text.
            log.error("supervisor.pipeline_failure", error=str(exc), exc_info=True)
            state.record("supervisor", ok=False, detail=f"error: {exc}")
            return QueryResponse(
                status=Status.ESCALATED,
                trace_id=state.trace_id,
                abstention_reason="pipeline failure; routed to human review",
                steps_run=[s.agent for s in state.steps],
            )
        finally:
            clear_run()

    # -- pipeline ----------------------------------------------------------
    def _run(
        self,
        request: QueryRequest,
        state: WorkflowState,
        history: list[tuple[str, str]] | None = None,
    ) -> QueryResponse:
        plan = self.planner.run(request, history)
        state.record("planner", ok=True, detail=plan.task_type.value)

        retrieved: list[RetrievedAuthority] = self.retrieve(plan.search_queries)
        state.record("retrieval", ok=bool(retrieved), detail=f"{len(retrieved)} authorities")

        if not retrieved:
            return self._abstain(state, plan.task_type, "no authority retrieved for the request")

        draft = self.drafting.run(request.question, retrieved)
        state.record("drafting", ok=bool(draft.claims), detail=f"{len(draft.claims)} claims")

        report = validate(draft, retrieved, threshold=self._threshold)
        verified = [v for v in report.verdicts if v.citation_exists and v.supported]
        state.record(
            "validation",
            ok=bool(verified),
            detail=f"{len(verified)} verified / {len(report.hallucinated_citations)} hallucinated",
        )

        # --- grounding policy ---------------------------------------------
        if not verified:
            reason = (
                "drafted citations were hallucinated or unsupported; abstaining rather than "
                "asserting ungrounded law"
                if draft.claims
                else "insufficient authority to ground an answer"
            )
            return self._abstain(state, plan.task_type, reason)

        response = self._compose(state, plan.task_type, verified, report, retrieved)
        log.info(
            "supervisor.done",
            status=response.status.value,
            confidence=response.confidence,
            citations=len(response.citations),
        )
        return response

    # -- helpers -----------------------------------------------------------
    def _compose(
        self,
        state: WorkflowState,
        task_type: TaskType,
        verified: list[ClaimVerdict],
        report: ValidationReport,
        retrieved: list[RetrievedAuthority],
    ) -> QueryResponse:
        by_id = {r.authority.id: r.authority for r in retrieved}
        answer = " ".join(v.claim.text for v in verified)
        citations = [
            CitationOut(
                citation=by_id[v.claim.authority_id].citation,
                authority_id=v.claim.authority_id,
                quote=v.claim.quote,
                source_url=by_id[v.claim.authority_id].source_url,
            )
            for v in verified
        ]
        confidence = round(sum(v.support_score for v in verified) / len(verified), 4)

        # High-risk signals -> human review, but still return the verified answer.
        had_hallucination = bool(report.hallucinated_citations)
        status = Status.ANSWERED
        reason = None
        if confidence < _LOW_CONFIDENCE or had_hallucination:
            status = Status.ESCALATED
            reason = (
                "low grounding confidence"
                if confidence < _LOW_CONFIDENCE
                else "drafting attempted a hallucinated citation; verified subset returned "
                "for review"
            )

        return QueryResponse(
            status=status,
            trace_id=state.trace_id,
            task_type=task_type,
            answer=answer,
            citations=citations,
            confidence=confidence,
            abstention_reason=reason,
            steps_run=[s.agent for s in state.steps],
        )

    def _abstain(self, state: WorkflowState, task_type: TaskType, reason: str) -> QueryResponse:
        log.info("supervisor.abstain", reason=reason)
        return QueryResponse(
            status=Status.ABSTAINED,
            trace_id=state.trace_id,
            task_type=task_type,
            abstention_reason=reason,
            confidence=0.0,
            steps_run=[s.agent for s in state.steps],
        )


def build_supervisor(settings: Settings) -> Supervisor:
    """Assemble the full pipeline from settings. Used by the API, CLI, and eval."""
    model = build_model(settings)
    corpus = load_corpus(settings)
    retriever = HybridRetriever(corpus, settings)
    return Supervisor(
        planner=PlannerAgent(model),
        retrieve=make_retrieve_tool(retriever),
        drafting=DraftingAgent(model),
        grounding_threshold=settings.grounding_threshold,
    )
