"""Agent base class.

Each agent has a well-defined interface (typed input -> typed output) and a
name used in logs and tracing, so a failure is attributable to a specific agent
rather than one opaque prompt chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from legalintel.logging import get_logger

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Agent(ABC, Generic[TIn, TOut]):
    name: str = "agent"

    def __init__(self) -> None:
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def run(self, inp: TIn) -> TOut:  # pragma: no cover - interface
        ...
