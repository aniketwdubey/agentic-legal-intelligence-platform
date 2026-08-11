"""Golden-set format and loader.

A golden item pairs a question with its reference answer, the authority ids that
*should* support it, and whether the correct behaviour is to abstain (used to
score the abstention path). Stored as JSONL so items are easy to append and diff
in review.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    id: str
    question: str
    jurisdiction: str | None = None
    reference_answer: str = ""
    # Corpus ids of the authorities a correct answer must rely on.
    expected_authorities: list[str] = Field(default_factory=list)
    # True when there is deliberately no supporting authority in the corpus and
    # the system should abstain instead of answering.
    expect_abstention: bool = False


def load_golden(path: str | Path) -> list[GoldenItem]:
    p = Path(path)
    items: list[GoldenItem] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(GoldenItem.model_validate(json.loads(line)))
    return items
