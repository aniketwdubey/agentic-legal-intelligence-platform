"""Planner agent: interprets the request at runtime and decomposes it.

Intent-agnostic — there are no hardcoded per-intent pipelines. A Strands ``Agent``
returns a validated ``Plan`` (via structured output); the supervisor executes
whatever plan comes back. A fresh ``Agent`` is built per call so no conversation
state leaks between requests.
"""

from __future__ import annotations

import structlog
from strands import Agent
from strands.models import Model

from legalintel.agents.hooks import LoggingHook
from legalintel.agents.planner.prompt import PLANNER_SYSTEM, planner_prompt
from legalintel.agents.planner.schema import Plan, PlanStep
from legalintel.schemas import QueryRequest

log = structlog.get_logger("agent.planner")


class PlannerAgent:
    name = "planner"

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(self, request: QueryRequest) -> Plan:
        agent = Agent(
            model=self._model,
            system_prompt=PLANNER_SYSTEM,
            name=self.name,
            hooks=[LoggingHook()],
            callback_handler=None,  # no stdout streaming; we log via the hook
        )
        result = agent(
            planner_prompt(request.question, request.jurisdiction),
            structured_output_model=Plan,
        )
        plan = result.structured_output
        assert isinstance(plan, Plan)  # structured_output_model guarantees the type
        # Guardrail: every plan must end with a validate step, whatever the model proposed.
        if PlanStep.VALIDATE not in plan.steps:
            plan = plan.model_copy(update={"steps": [*plan.steps, PlanStep.VALIDATE]})
        log.info(
            "planner.plan",
            task_type=plan.task_type.value,
            steps=[s.value for s in plan.steps],
            queries=plan.search_queries,
        )
        return plan
