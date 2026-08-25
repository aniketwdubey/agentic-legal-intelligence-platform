"""Amazon Bedrock AgentCore Runtime entrypoint.

Wraps the same citation-verified supervisor pipeline behind the AgentCore Runtime
HTTP contract (`BedrockAgentCoreApp` serves `/ping` + `/invocations` on port 8080).
The one capability this adds over the Lambda deployment is **multi-turn
conversation**: each session's recent turns are pulled from AgentCore Memory and
fed to the planner so follow-up questions resolve, then the new turn is stored.

Invocation payload:  {"prompt": "<legal question>"}
Response:            the QueryResponse JSON (status, answer, verified citations, ...)

Memory is optional — set LEGALINTEL_MEMORY_ID to enable it. Without it, the agent
still works as a stateless single-shot (identical to the Lambda behaviour).
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from legalintel.agents import build_supervisor
from legalintel.config import get_settings
from legalintel.schemas import QueryRequest

log = structlog.get_logger("agentcore")

_settings = get_settings()
# Build the pipeline once (corpus load + index) and reuse across invocations.
_supervisor = build_supervisor(_settings)

_MEMORY_ID = os.environ.get("LEGALINTEL_MEMORY_ID") or None
_ACTOR_ID = os.environ.get("LEGALINTEL_MEMORY_ACTOR", "legal-user")
_K_TURNS = int(os.environ.get("LEGALINTEL_MEMORY_TURNS", "6"))

_memory: Any = None


def _mem() -> Any:
    """Lazily build the Memory client (only when memory is configured)."""
    global _memory
    if _memory is None and _MEMORY_ID:
        from bedrock_agentcore.memory import MemoryClient

        _memory = MemoryClient(region_name=_settings.aws_region)
    return _memory


app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any) -> dict[str, Any]:
    question = (payload or {}).get("prompt", "")
    if not isinstance(question, str) or not question.strip():
        return {"status": "error", "error": "request must include a non-empty 'prompt'"}
    question = question.strip()
    session_id = getattr(context, "session_id", None) or "default-session"

    history = _load_history(session_id)
    resp = _supervisor.run(QueryRequest(question=question), history=history)
    _save_turn(session_id, question, resp.answer or resp.abstention_reason or "")

    return dict(resp.model_dump(mode="json"))


# --------------------------------------------------------------------------
# AgentCore Memory (short-term, per-session conversation)
# --------------------------------------------------------------------------
def _load_history(session_id: str) -> list[tuple[str, str]] | None:
    mem = _mem()
    if not mem:
        return None
    try:
        turns = mem.get_last_k_turns(
            memory_id=_MEMORY_ID, actor_id=_ACTOR_ID, session_id=session_id, k=_K_TURNS
        )
    except Exception as exc:  # memory is best-effort — never fail the answer over it
        log.warning("memory.load_failed", error=str(exc))
        return None

    out: list[tuple[str, str]] = []
    for turn in turns:
        for msg in turn:
            role = str(msg.get("role", "")).upper()
            text = _text_of(msg)
            if text:
                out.append((role, text))
    return out or None


def _save_turn(session_id: str, question: str, answer: str) -> None:
    mem = _mem()
    if not mem:
        return
    try:
        mem.create_event(
            memory_id=_MEMORY_ID,
            actor_id=_ACTOR_ID,
            session_id=session_id,
            messages=[(question, "USER"), (answer or "(abstained)", "ASSISTANT")],
        )
    except Exception as exc:
        log.warning("memory.save_failed", error=str(exc))


def _text_of(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, str):
        return content
    return ""


if __name__ == "__main__":
    # Serves /invocations + /ping on 0.0.0.0:8080 (the AgentCore Runtime contract).
    app.run()
