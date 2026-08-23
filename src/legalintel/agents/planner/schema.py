"""Planner output schema — the interpreted intent for a query."""

from __future__ import annotations

from pydantic import BaseModel, Field

from legalintel.schemas import TaskType


class Plan(BaseModel):
    """Planner output: the interpreted intent that shapes retrieval.

    The execution pipeline (retrieve -> draft -> validate) is a fixed, deterministic
    sequence run by the supervisor, so the plan describes *what* to look for
    (task_type, queries), not *how* the steps run.
    """

    task_type: TaskType
    jurisdiction: str = "US"
    search_queries: list[str] = Field(
        min_length=1, description="Queries the retrieval agent should issue."
    )
    rationale: str = Field(description="Why the planner interpreted the request this way.")
