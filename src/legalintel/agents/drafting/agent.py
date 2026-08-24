"""Drafting agent: writes a grounded answer from retrieved authority only.

Produces a ``DraftAnswer`` whose every claim references a provided authority. The
output is *not trusted yet* — the validator independently verifies each citation
before anything reaches the user. A fresh ``Agent`` is built per call so no
conversation state leaks between requests.
"""

from __future__ import annotations

import structlog
from strands import Agent
from strands.models import Model

from legalintel.agents.drafting.prompt import DRAFTER_SYSTEM, drafter_prompt
from legalintel.agents.drafting.schema import DraftAnswer
from legalintel.agents.errors import GuardrailBlocked
from legalintel.agents.hooks import LoggingHook
from legalintel.schemas import RetrievedAuthority

log = structlog.get_logger("agent.drafting")


class DraftingAgent:
    name = "drafting"

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(self, question: str, retrieved: list[RetrievedAuthority]) -> DraftAnswer:
        if not retrieved:
            # Nothing to ground on -> honest empty draft (supervisor abstains).
            log.info("drafting.no_context")
            return DraftAnswer(answer="", claims=[])
        agent = Agent(
            model=self._model,
            system_prompt=DRAFTER_SYSTEM,
            name=self.name,
            hooks=[LoggingHook()],
            callback_handler=None,  # no stdout streaming; we log via the hook
        )
        result = agent(
            drafter_prompt(question, retrieved),
            structured_output_model=DraftAnswer,
        )
        if result.stop_reason in ("guardrail_intervened", "content_filtered"):
            raise GuardrailBlocked(
                f"drafter output blocked by the safety guardrail ({result.stop_reason})"
            )
        draft = result.structured_output
        if not isinstance(draft, DraftAnswer):  # no tool call produced (blocked or refused)
            raise GuardrailBlocked(f"drafter produced no answer (stop_reason={result.stop_reason})")
        log.info("drafting.done", claims=len(draft.claims))
        return draft
