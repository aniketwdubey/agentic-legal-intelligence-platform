"""LLM guardrails: schema validation + retry-with-backoff in `structured()`."""

from __future__ import annotations

import pytest

from legalintel.config import Settings
from legalintel.llm.base import LLMClient, LLMTransientError, structured
from legalintel.schemas import Plan


class _FlakyClient(LLMClient):
    """Returns malformed JSON N times, then a valid Plan."""

    def __init__(self, bad_times: int, good_payload: str) -> None:
        self.calls = 0
        self.bad_times = bad_times
        self.good_payload = good_payload

    def complete_json(self, *, system, prompt, timeout):
        self.calls += 1
        if self.calls <= self.bad_times:
            return "not json at all"
        return self.good_payload


_VALID_PLAN = (
    '{"task_type": "research", "jurisdiction": "US", "search_queries": ["x"], '
    '"steps": ["retrieve", "validate"], "rationale": "ok"}'
)


def test_structured_retries_then_succeeds():
    settings = Settings(llm_max_retries=3)
    client = _FlakyClient(bad_times=2, good_payload=_VALID_PLAN)
    plan = structured(client, settings=settings, system="ROLE:PLANNER", prompt="p", schema=Plan)
    assert plan.task_type.value == "research"
    assert client.calls == 3  # 2 failures + 1 success


def test_structured_gives_up_after_max_retries():
    settings = Settings(llm_max_retries=1)
    client = _FlakyClient(bad_times=99, good_payload=_VALID_PLAN)
    with pytest.raises(LLMTransientError):
        structured(client, settings=settings, system="ROLE:PLANNER", prompt="p", schema=Plan)
    assert client.calls == 2  # initial + 1 retry


def test_timeout_is_passed_through_to_client():
    settings = Settings(llm_timeout_seconds=7.5)
    seen = {}

    class _Recorder(LLMClient):
        def complete_json(self, *, system, prompt, timeout):
            seen["timeout"] = timeout
            return _VALID_PLAN

    structured(_Recorder(), settings=settings, system="ROLE:PLANNER", prompt="p", schema=Plan)
    assert seen["timeout"] == 7.5
