"""Deterministic, offline Strands model provider for tests, CI, and demos.

Implements the Strands ``Model`` interface but makes **no network calls** — this
is what lets the whole agent pipeline (and every test) run without AWS. It is the
Strands-era replacement for the old ``llm/mock.py`` client.

It emulates a structured-output tool call: when the agent forces a pydantic tool
(``Plan`` or ``DraftAnswer``), ``stream`` yields the same Bedrock-style event
sequence a real provider would, with the tool ``input`` filled from a deterministic,
honestly-grounded builder. This keeps the mock on the **same, non-deprecated agent
code path** as the live Bedrock model (so hooks fire and behaviour matches).

The per-schema builders (``_plan`` / ``_draft``) are overridable so test doubles can
simulate hallucinated citations or model failures with a one-method subclass.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from strands.models import Model

from legalintel.agents.drafting.schema import Claim, DraftAnswer
from legalintel.agents.planner.schema import Plan

_DRAFT_KEYWORDS = ("draft", "write", "prepare", "nda", "agreement", "clause")
_REVIEW_KEYWORDS = ("review", "flag", "risky", "redline", "risk")


class StubModel(Model):
    """Offline model: deterministic structured output, no network."""

    def get_config(self) -> dict[str, Any]:
        return {"model_id": "stub"}

    def update_config(self, **model_config: Any) -> None:  # pragma: no cover - no-op
        return None

    async def structured_output(
        self, output_model: type[Any], prompt: Any = None, system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Supports the (deprecated) direct path too, for completeness.
        instance = self._build(output_model.__name__, _flatten(prompt))
        yield {"output": instance}

    async def stream(  # type: ignore[override]  # returns Bedrock-style event dicts
        self,
        messages: Any,
        tool_specs: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Emulate a forced structured-output tool call as a Bedrock-style stream."""
        if not tool_specs:
            raise NotImplementedError("StubModel only supports structured-output tool calls")
        # Pick the structured-output tool by name, so an unrelated tool attached to
        # the agent doesn't shadow it at index 0.
        spec = next(
            (s for s in tool_specs if s.get("name") in ("Plan", "DraftAnswer")), tool_specs[0]
        )
        name = spec["name"]
        instance = self._build(name, _flatten(messages))
        payload = json.dumps(instance.model_dump(mode="json"))

        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "stub-1", "name": name}},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": payload}},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }

    # -- structured-instance builders (overridable in test doubles) ---------
    def _build(self, model_name: str, text: str) -> Any:
        if model_name == "Plan":
            return self._plan(text)
        if model_name == "DraftAnswer":
            return self._draft(text)
        raise NotImplementedError(f"StubModel has no builder for {model_name!r}")

    def _plan(self, text: str) -> Plan:
        question = _extract("QUESTION", text)
        jur = _extract("JURISDICTION_HINT", text) or "US"
        q = question.lower()
        if _has_keyword(q, _DRAFT_KEYWORDS):
            task_type = "drafting"
        elif _has_keyword(q, _REVIEW_KEYWORDS):
            task_type = "doc_review"
        else:
            task_type = "research"
        return Plan(
            task_type=task_type,  # type: ignore[arg-type]  # str coerces to TaskType
            jurisdiction=jur if jur != "unknown" else "US",
            search_queries=[question or text],
            rationale=f"Interpreted request as {task_type}; retrieve, then ground and verify.",
        )

    def _draft(self, text: str) -> DraftAnswer:
        question = _extract("QUESTION", text)
        raw_ctx = _extract("CONTEXT_JSON", text)
        try:
            context = json.loads(raw_ctx) if raw_ctx else []
        except json.JSONDecodeError:
            context = []
        if not context:
            # No authority to ground on -> honest empty draft -> triggers abstention.
            return DraftAnswer(answer="", claims=[])

        claims: list[Claim] = []
        # Ground on the single best-matching authority (deterministic happy path).
        for item in context[:1]:
            quote = _first_sentence(item.get("text", ""))
            if not quote:
                continue
            claims.append(
                Claim(
                    text=(
                        f"Regarding '{question}', {item['citation']} "
                        "provides applicable authority."
                    ),
                    authority_id=item["id"],
                    quote=quote,
                )
            )
        answer = " ".join(c.text for c in claims)
        return DraftAnswer(answer=answer, claims=claims)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _flatten(prompt: Any) -> str:
    """Concatenate all text blocks of a Strands message list into one string."""
    if isinstance(prompt, str):
        return prompt
    parts: list[str] = []
    for msg in prompt or []:
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def _extract(marker: str, text: str) -> str:
    """Return the text following ``MARKER:`` up to the end of its line."""
    m = re.search(rf"{re.escape(marker)}:\s*(.*)", text)
    return m.group(1).strip() if m else ""


def _has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Whole-word match so e.g. 'nda' doesn't fire inside 'sta*nda*rd'."""
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def _first_sentence(text: str, limit: int = 240) -> str:
    text = text.strip()
    m = re.search(r"(.+?[.;])\s", text + " ")
    span = m.group(1) if m else text
    return span[:limit].strip()
