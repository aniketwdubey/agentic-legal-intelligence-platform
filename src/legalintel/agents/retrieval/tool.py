"""Retrieval, expressed as a Strands ``@tool``.

Retrieval is a deterministic capability, not an LLM agent, so it is modelled as a
tool the pipeline (and, in future, an orchestrator agent) can call. It is decoupled
from the planner: it takes a plain ``list[str]`` of queries — not a ``Plan`` — so it
shares nothing but data with the rest of the graph.

``make_retrieve_tool`` binds a concrete ``HybridRetriever`` and returns the tool.
The supervisor invokes it directly (a decorated tool called with normal args returns
its normal value); attaching it to an agent's ``tools=[...]`` would let a model call
it too.
"""

from __future__ import annotations

from typing import Any

import structlog
from strands import tool

from legalintel.retrieval.hybrid import HybridRetriever
from legalintel.schemas import RetrievedAuthority

log = structlog.get_logger("agent.retrieval")


def make_retrieve_tool(retriever: HybridRetriever) -> Any:
    """Return a Strands ``retrieve`` tool bound to ``retriever``."""

    @tool
    def retrieve(queries: list[str]) -> list[RetrievedAuthority]:
        """Retrieve the most relevant legal authorities for the given search queries.

        Args:
            queries: One or more search strings to look up in the legal corpus.

        Returns:
            The fused top-k authorities (deduplicated across queries), each scored.
        """
        results = retriever.search_many(queries)
        log.info(
            "retrieval.done",
            authorities=[r.authority.id for r in results],
            count=len(results),
        )
        return results

    return retrieve
