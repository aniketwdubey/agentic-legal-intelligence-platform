"""HTTP routes — thin handlers that delegate to the supervisor.

No orchestration or grounding logic here; the handler validates input via the
request schema, calls the supervisor, and returns its typed response.
"""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from legalintel.agents import Supervisor
from legalintel.schemas import QueryRequest, QueryResponse

router = APIRouter()


def _supervisor(request: Request) -> Supervisor:
    return cast(Supervisor, request.app.state.supervisor)


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/query", response_model=QueryResponse, tags=["query"])
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Answer a legal question with verified citations, or abstain/escalate."""
    return _supervisor(request).run(payload)
