"""Drafting prompt.

The user prompt embeds the retrieved authorities as ``CONTEXT_JSON:`` so the
model grounds ONLY in that set (and the offline ``StubModel`` can parse it).
Strands enforces the ``DraftAnswer`` schema via its structured-output tool spec.
"""

from __future__ import annotations

import json

from legalintel.schemas import RetrievedAuthority

DRAFTER_SYSTEM = """You are the drafting agent. Using ONLY the retrieved authorities
provided, write a grounded answer. Every legal proposition you assert MUST be a claim
whose authority_id is one of the provided authority ids and whose quote is a verbatim
span copied from that authority's text. Never cite an authority that is not in the
provided context. If the context is insufficient, return an empty claims list."""


def drafter_prompt(question: str, retrieved: list[RetrievedAuthority]) -> str:
    context = [
        {"id": r.authority.id, "citation": r.authority.citation, "text": r.authority.text}
        for r in retrieved
    ]
    return (
        f"QUESTION: {question}\n"
        f"CONTEXT_JSON: {json.dumps(context)}\n"
        "Write the grounded answer."
    )
