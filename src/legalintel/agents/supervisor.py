"""Supervisor control layer — orchestration, grounding policy, recovery.

Runs the plan's steps against shared workflow state, validates every agent
output at the boundary, and makes the final answer/abstain/escalate decision.
The core discipline: **a claim reaches the user only if the validation agent
verified both that its citation exists in the retrieved set and that the cited
authority supports it.** Anything less becomes an abstention or a human-review
escalation — the platform never invents law to fill a gap.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from legalintel.agents.drafting import DraftingAgent, DraftInput
from legalintel.agents.planner import PlannerAgent
from legalintel.agents.retrieval import RetrievalAgent
from legalintel.agents.validation import ValidationAgent, ValidationInput
from legalintel.config import Settings
from legalintel.llm import LLMError, build_llm_client
from legalintel.logging import bind_run, clear_run, get_logger
from legalintel.retrieval import HybridRetriever, load_corpus
from legalintel.schemas import (
    CitationOut,
    ClaimVerdict,
    QueryRequest,
    QueryResponse,
    RetrievedAuthority,
    Status,
    TaskType,
    ValidationReport,
)

log = get_logger("supervisor")

# Below this mean support score we escalate rather than answer outright.
_LOW_CONFIDENCE = 0.35


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
        retrieval: RetrievalAgent,
        drafting: DraftingAgent,
        validation: ValidationAgent,
    ) -> None:
        self.planner = planner
        self.retrieval = retrieval
        self.drafting = drafting
        self.validation = validation

    def run(self, request: QueryRequest) -> QueryResponse:
        state = WorkflowState(trace_id=uuid.uuid4().hex[:12])
        bind_run(trace_id=state.trace_id)
        try:
            return self._run(request, state)
        except LLMError as exc:
            # Non-retryable model failure after guardrail retries: escalate, never guess.
            log.error("supervisor.llm_failure", error=str(exc))
            state.record("supervisor", ok=False, detail=f"llm_error: {exc}")
            return QueryResponse(
                status=Status.ESCALATED,
                trace_id=state.trace_id,
                abstention_reason="model unavailable after retries; routed to human review",
                steps_run=[s.agent for s in state.steps],
            )
        finally:
            clear_run()

    # -- pipeline ----------------------------------------------------------
    def _run(self, request: QueryRequest, state: WorkflowState) -> QueryResponse:
        plan = self.planner.run(request)
        state.record("planner", ok=True, detail=plan.task_type.value)

        retrieved = self.retrieval.run(plan)
        state.record("retrieval", ok=bool(retrieved), detail=f"{len(retrieved)} authorities")

        if not retrieved:
            return self._abstain(state, plan.task_type, "no authority retrieved for the request")

        draft = self.drafting.run(DraftInput(question=request.question, retrieved=retrieved))
        state.record("drafting", ok=bool(draft.claims), detail=f"{len(draft.claims)} claims")

        report = self.validation.run(ValidationInput(draft=draft, retrieved=retrieved))
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
    client = build_llm_client(settings)
    corpus = load_corpus(settings)
    retriever = HybridRetriever(corpus, settings)
    return Supervisor(
        planner=PlannerAgent(client, settings),
        retrieval=RetrievalAgent(retriever),
        drafting=DraftingAgent(client, settings),
        validation=ValidationAgent(settings),
    )
