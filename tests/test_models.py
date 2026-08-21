"""Model factory + offline StubModel behaviour (no network)."""

from __future__ import annotations

from strands import Agent
from strands.models import BedrockModel

from legalintel.agents._stub_model import StubModel
from legalintel.agents.drafting.prompt import DRAFTER_SYSTEM, drafter_prompt
from legalintel.agents.drafting.schema import DraftAnswer
from legalintel.agents.planner.prompt import PLANNER_SYSTEM, planner_prompt
from legalintel.agents.planner.schema import Plan
from legalintel.config import Settings
from legalintel.models import build_model
from legalintel.schemas import Authority, AuthorityType, RetrievedAuthority


def test_build_model_mock_is_offline_stub():
    assert isinstance(build_model(Settings(llm_provider="mock")), StubModel)


def test_build_model_bedrock_constructs_without_network():
    # Constructing the client resolves no credentials and makes no call.
    model = build_model(Settings(llm_provider="bedrock", aws_region="us-east-1"))
    assert isinstance(model, BedrockModel)


def test_stub_model_yields_valid_plan_through_agent():
    agent = Agent(model=StubModel(), system_prompt=PLANNER_SYSTEM, callback_handler=None)
    plan = agent(
        planner_prompt("What is the standard for fair use?", None),
        structured_output_model=Plan,
    ).structured_output
    assert isinstance(plan, Plan)
    assert plan.search_queries


def test_stub_model_grounds_draft_only_in_context():
    auth = Authority(
        id="fix-statute-a",
        citation="Test § 1",
        type=AuthorityType.STATUTE,
        title="Fair use",
        text="The fair use of a copyrighted work for teaching is not an infringement of copyright.",
    )
    agent = Agent(model=StubModel(), system_prompt=DRAFTER_SYSTEM, callback_handler=None)
    draft = agent(
        drafter_prompt("fair use?", [RetrievedAuthority(authority=auth, score=0.9)]),
        structured_output_model=DraftAnswer,
    ).structured_output
    assert isinstance(draft, DraftAnswer)
    assert draft.claims and draft.claims[0].authority_id == "fix-statute-a"
