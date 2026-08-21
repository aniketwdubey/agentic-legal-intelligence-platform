"""Planner output schema — the interpreted intent and ordered steps."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from legalintel.schemas import TaskType


class PlanStep(str, Enum):
    RETRIEVE = "retrieve"
    DRAFT = "draft"
    VALIDATE = "validate"


class Plan(BaseModel):
    """Planner output: the interpreted intent and the ordered steps to run."""

    task_type: TaskType
    jurisdiction: str = "US"
    search_queries: list[str] = Field(
        min_length=1, description="Queries the retrieval agent should issue."
    )
    steps: list[PlanStep] = Field(min_length=1)
    rationale: str = Field(description="Why the planner chose these steps.")
