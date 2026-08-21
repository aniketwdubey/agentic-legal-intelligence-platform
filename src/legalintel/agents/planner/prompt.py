"""Planner prompt.

The user prompt embeds machine-readable markers (``QUESTION:``,
``JURISDICTION_HINT:``) so the offline ``StubModel`` can parse it deterministically;
the live model simply reads them as context. Strands enforces the ``Plan`` schema
via its structured-output tool spec, so the prompt does not restate the schema.
"""

from __future__ import annotations

PLANNER_SYSTEM = """You are the planning agent of a legal-intelligence platform.
Interpret the user's request at runtime and decompose it. Choose task_type from
[research, drafting, doc_review], the jurisdiction, one or more search_queries to
retrieve authority, and the ordered steps from [retrieve, draft, validate]. Every
task must end with a validate step. Do not assert any legal proposition here."""


def planner_prompt(question: str, jurisdiction: str | None) -> str:
    return (
        f"QUESTION: {question}\n"
        f"JURISDICTION_HINT: {jurisdiction or 'unknown'}\n"
        "Produce the plan."
    )
