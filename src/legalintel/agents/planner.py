"""Planner agent: interprets the request at runtime and decomposes it.

Intent-agnostic — there are no hardcoded per-intent pipelines. The planner
chooses the task type, search queries, and the ordered steps; the supervisor
executes whatever plan comes back (after schema validation).
"""

from __future__ import annotations

from legalintel.agents.base import Agent
from legalintel.config import Settings
from legalintel.llm import LLMClient, structured
from legalintel.llm.prompts import PLANNER_SYSTEM, planner_prompt
from legalintel.schemas import Plan, PlanStep, QueryRequest


class PlannerAgent(Agent[QueryRequest, Plan]):
    name = "planner"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    def run(self, inp: QueryRequest) -> Plan:
        plan = structured(
            self._client,
            settings=self._settings,
            system=PLANNER_SYSTEM,
            prompt=planner_prompt(inp.question, inp.jurisdiction),
            schema=Plan,
        )
        # Guardrail: every plan must retrieve before drafting and must validate,
        # regardless of what the model proposed.
        if PlanStep.VALIDATE not in plan.steps:
            plan = plan.model_copy(update={"steps": [*plan.steps, PlanStep.VALIDATE]})
        self.log.info(
            "planner.plan",
            task_type=plan.task_type.value,
            steps=[s.value for s in plan.steps],
            queries=plan.search_queries,
        )
        return plan
