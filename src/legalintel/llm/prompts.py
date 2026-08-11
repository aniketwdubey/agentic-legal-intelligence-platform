"""Prompt construction for the LLM-backed agents (planner, drafter).

Prompts embed machine-readable markers (``ROLE:``, ``QUESTION:``,
``CONTEXT_JSON:``) so the deterministic mock client can parse them offline and
so real completions stay easy to debug. Only the planner and drafter call the
LLM; validation is rule-based (see ``agents/validation.py``).
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from legalintel.schemas import DraftAnswer, Plan, RetrievedAuthority

ROLE_PLANNER = "ROLE:PLANNER"
ROLE_DRAFTER = "ROLE:DRAFTER"


def _schema_hint(model: type[BaseModel]) -> str:
    """Render a compact JSON-Schema block so a real model emits conformant fields.

    The offline mock parses prompt markers instead, so this only guides live
    providers; without it a real completion tends to drop required fields
    (e.g. Plan.rationale) and fail schema validation in ``structured()``.
    """
    return json.dumps(model.model_json_schema(), separators=(",", ":"))


PLANNER_SYSTEM = f"""{ROLE_PLANNER}
You are the planning agent of a legal-intelligence platform. Interpret the user's
request at runtime and decompose it. Choose task_type from
[research, drafting, doc_review], the jurisdiction, one or more search_queries to
retrieve authority, and the ordered steps from [retrieve, draft, validate].
Every task must end with a validate step. Do not assert any legal proposition here.
Respond with ONLY a JSON object conforming to this JSON Schema (include every
required field):
{_schema_hint(Plan)}"""

DRAFTER_SYSTEM = f"""{ROLE_DRAFTER}
You are the drafting agent. Using ONLY the retrieved authorities provided, write a
grounded answer. Every legal proposition you assert MUST be a claim whose
authority_id is one of the provided authority ids and whose quote is a verbatim
span copied from that authority's text. Never cite an authority that is not in the
provided context. If the context is insufficient, return an empty claims list.
Respond with ONLY a JSON object conforming to this JSON Schema (include every
required field):
{_schema_hint(DraftAnswer)}"""


def planner_prompt(question: str, jurisdiction: str | None) -> str:
    return (
        f"QUESTION: {question}\n"
        f"JURISDICTION_HINT: {jurisdiction or 'unknown'}\n"
        "Return the Plan JSON now."
    )


def drafter_prompt(question: str, retrieved: list[RetrievedAuthority]) -> str:
    context = [
        {"id": r.authority.id, "citation": r.authority.citation, "text": r.authority.text}
        for r in retrieved
    ]
    return (
        f"QUESTION: {question}\n"
        f"CONTEXT_JSON: {json.dumps(context)}\n"
        "Write the grounded DraftAnswer JSON now."
    )
