"""Tiny CLI to ask the platform a single question and print the grounded result.

    python scripts/ask.py "What is the standard for a motion to dismiss?"

Uses the configured provider (mock by default; set LEGALINTEL_LLM_PROVIDER=bedrock
and AWS creds for a live run).
"""

from __future__ import annotations

import sys

from legalintel.agents import build_supervisor
from legalintel.config import get_settings
from legalintel.logging import configure_logging
from legalintel.schemas import QueryRequest


def main(argv: list[str]) -> int:
    if not argv:
        print('usage: python scripts/ask.py "your legal question"')
        return 2

    settings = get_settings()
    configure_logging(settings)
    supervisor = build_supervisor(settings)
    resp = supervisor.run(QueryRequest(question=" ".join(argv)))

    provider_line = settings.llm_provider + (
        f" ({settings.bedrock_model_id} @ {settings.aws_region})"
        if settings.llm_provider == "bedrock"
        else " (offline)"
    )
    print(f"\nprovider   : {provider_line}")
    print(f"status     : {resp.status.value}")
    print(f"trace_id   : {resp.trace_id}")
    print(f"task_type  : {resp.task_type.value if resp.task_type else '-'}")
    print(f"confidence : {resp.confidence}")
    if resp.abstention_reason:
        print(f"note       : {resp.abstention_reason}")
    if resp.answer:
        print(f"\nanswer:\n{resp.answer}")
    if resp.citations:
        print("\ncitations (verified):")
        for c in resp.citations:
            print(f"  - {c.citation}  [{c.authority_id}]")
            if c.quote:
                print(f'      "{c.quote}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
