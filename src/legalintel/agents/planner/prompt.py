"""Planner prompt.

The user prompt embeds machine-readable markers (``QUESTION:``,
``JURISDICTION_HINT:``) so the offline ``StubModel`` can parse it deterministically;
the live model simply reads them as context. Strands enforces the ``Plan`` schema
via its structured-output tool spec, so the prompt does not restate the schema.
"""

from __future__ import annotations

PLANNER_SYSTEM = """You are the planning agent of a legal-intelligence platform.
Interpret the user's request at runtime. Choose task_type from
[research, drafting, doc_review], the jurisdiction, and one or more search_queries
that will retrieve the authority needed to answer it. Do not assert any legal
proposition here."""


def planner_prompt(
    question: str,
    jurisdiction: str | None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    convo = ""
    if history:
        lines = "\n".join(f"{role}: {text}" for role, text in history)
        convo = (
            "CONVERSATION so far (resolve any follow-up references in the question "
            f"against this history):\n{lines}\n"
        )
    return (
        f"{convo}QUESTION: {question}\n"
        f"JURISDICTION_HINT: {jurisdiction or 'unknown'}\n"
        "Produce the plan."
    )
