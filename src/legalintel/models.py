"""Model provider factory.

Returns the Strands model the agents run on, selected by ``settings.llm_provider``:

* ``bedrock`` -> ``strands.models.BedrockModel`` (Amazon Bedrock, pay-per-call).
  Credentials come from the standard AWS chain; none are read here.
* ``mock``    -> an offline ``StubModel`` (deterministic, zero-cost) — the default
  and what tests/CI use.

Retries and schema validation are handled by Strands itself, so there is no
bespoke guarded ``structured()`` helper any more; the request read-timeout is the
one reliability knob we still thread through to Bedrock.
"""

from __future__ import annotations

from typing import Any

from strands.models import Model

from legalintel.config import Settings


def build_model(settings: Settings) -> Model:
    """Return the configured Strands model. ``mock`` needs no AWS deps."""
    if settings.llm_provider == "bedrock":
        from botocore.config import Config
        from strands.models import BedrockModel

        kwargs: dict[str, Any] = {
            "model_id": settings.bedrock_model_id,
            "region_name": settings.aws_region,
            "max_tokens": settings.bedrock_max_tokens,
            "boto_client_config": Config(read_timeout=int(settings.llm_timeout_seconds) + 5),
        }
        # Apply the Bedrock Guardrail (prompt-injection + PII) on every call when set.
        if settings.bedrock_guardrail_id:
            kwargs["guardrail_id"] = settings.bedrock_guardrail_id
            kwargs["guardrail_version"] = settings.bedrock_guardrail_version or "DRAFT"
            kwargs["guardrail_trace"] = "enabled"
        return BedrockModel(**kwargs)

    # Lazy import keeps the mock path free of any Strands-Bedrock/boto surface and
    # avoids an import cycle (the stub imports agent schemas).
    from legalintel.agents._stub_model import StubModel

    return StubModel()
