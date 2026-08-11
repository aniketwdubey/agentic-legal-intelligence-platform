"""Retrieval agent: fetches candidate authorities for the plan's queries.

Thin agent wrapper around the hybrid retriever so the supervisor treats
retrieval like any other typed agent step (and so it can later be swapped for a
tool call to an external search service without touching the orchestration).
"""

from __future__ import annotations

from legalintel.agents.base import Agent
from legalintel.retrieval.hybrid import HybridRetriever
from legalintel.schemas import Plan, RetrievedAuthority


class RetrievalAgent(Agent[Plan, list[RetrievedAuthority]]):
    name = "retrieval"

    def __init__(self, retriever: HybridRetriever) -> None:
        super().__init__()
        self._retriever = retriever

    def run(self, inp: Plan) -> list[RetrievedAuthority]:
        results = self._retriever.search_many(inp.search_queries)
        self.log.info(
            "retrieval.done",
            authorities=[r.authority.id for r in results],
            count=len(results),
        )
        return results
