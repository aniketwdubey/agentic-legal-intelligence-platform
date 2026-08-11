"""Drafting agent: writes a grounded answer/draft from retrieved authority only.

Produces a ``DraftAnswer`` whose every claim references a retrieved authority.
The output is *not trusted yet* — it is handed to the validation agent, which
independently verifies each citation before anything reaches the user.
"""

from __future__ import annotations

from dataclasses import dataclass

from legalintel.agents.base import Agent
from legalintel.config import Settings
from legalintel.llm import LLMClient, structured
from legalintel.llm.prompts import DRAFTER_SYSTEM, drafter_prompt
from legalintel.schemas import DraftAnswer, RetrievedAuthority


@dataclass
class DraftInput:
    question: str
    retrieved: list[RetrievedAuthority]


class DraftingAgent(Agent[DraftInput, DraftAnswer]):
    name = "drafting"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    def run(self, inp: DraftInput) -> DraftAnswer:
        if not inp.retrieved:
            # Nothing to ground on -> honest empty draft (supervisor abstains).
            self.log.info("drafting.no_context")
            return DraftAnswer(answer="", claims=[])
        draft = structured(
            self._client,
            settings=self._settings,
            system=DRAFTER_SYSTEM,
            prompt=drafter_prompt(inp.question, inp.retrieved),
            schema=DraftAnswer,
        )
        self.log.info("drafting.done", claims=len(draft.claims))
        return draft
