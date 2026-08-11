"""Deterministic, offline mock LLM client.

Parses the prompt markers defined in ``prompts.py`` and returns well-formed,
schema-valid JSON with **no network calls** — this is what lets the full agent
pipeline, tests, and CI run without AWS credentials. It is intentionally
"honest": the drafter mock only ever cites authority ids present in the
retrieved context, so the happy path is grounded. Hallucination/abstention
paths are exercised in tests via purpose-built fake clients.
"""

from __future__ import annotations

import json
import re

from legalintel.llm.base import LLMClient
from legalintel.llm.prompts import ROLE_DRAFTER, ROLE_PLANNER

_DRAFT_KEYWORDS = ("draft", "write", "prepare", "nda", "agreement", "clause")
_REVIEW_KEYWORDS = ("review", "flag", "risky", "redline", "risk")


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word match so e.g. 'nda' doesn't fire inside 'sta*nda*rd'."""
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def _first_sentence(text: str, limit: int = 240) -> str:
    text = text.strip()
    m = re.search(r"(.+?[.;])\s", text + " ")
    span = m.group(1) if m else text
    return span[:limit].strip()


class MockLLMClient(LLMClient):
    def complete_json(self, *, system: str, prompt: str, timeout: float) -> str:
        if ROLE_PLANNER in system:
            return self._plan(prompt)
        if ROLE_DRAFTER in system:
            return self._draft(prompt)
        # Unknown role: return an empty object; the schema layer will reject it
        # and the caller's retry/abstention logic takes over.
        return "{}"

    # -- planner -----------------------------------------------------------
    def _plan(self, prompt: str) -> str:
        question = _extract("QUESTION", prompt)
        jur = _extract("JURISDICTION_HINT", prompt) or "US"
        q = question.lower()
        if _has_keyword(q, _DRAFT_KEYWORDS):
            task_type = "drafting"
        elif _has_keyword(q, _REVIEW_KEYWORDS):
            task_type = "doc_review"
        else:
            task_type = "research"
        return json.dumps(
            {
                "task_type": task_type,
                "jurisdiction": jur if jur != "unknown" else "US",
                "search_queries": [question],
                "steps": ["retrieve", "draft", "validate"],
                "rationale": f"Interpreted request as {task_type}; retrieve, then ground "
                "and verify.",
            }
        )

    # -- drafter -----------------------------------------------------------
    def _draft(self, prompt: str) -> str:
        question = _extract("QUESTION", prompt)
        raw_ctx = _extract("CONTEXT_JSON", prompt)
        try:
            context = json.loads(raw_ctx) if raw_ctx else []
        except json.JSONDecodeError:
            context = []

        if not context:
            # No authority to ground on -> honest empty draft -> triggers abstention.
            return json.dumps({"answer": "", "claims": []})

        claims = []
        # Ground on the single best-matching authority. A real drafting model
        # would compose a fuller answer; the mock stays deterministic and only
        # ever cites the top retrieved authority so the happy path is cleanly
        # grounded (multi-claim filtering is covered by the validation tests).
        for item in context[:1]:
            quote = _first_sentence(item.get("text", ""))
            if not quote:
                continue
            claim_text = (
                f"Regarding '{question}', {item['citation']} provides applicable authority."
            )
            claims.append(
                {
                    "text": claim_text,
                    "authority_id": item["id"],
                    "quote": quote,
                }
            )
        answer = " ".join(c["text"] for c in claims) or ""
        return json.dumps({"answer": answer, "claims": claims})


def _extract(marker: str, prompt: str) -> str:
    """Return the text following ``MARKER:`` up to the next line."""
    m = re.search(rf"{re.escape(marker)}:\s*(.*)", prompt)
    return m.group(1).strip() if m else ""
