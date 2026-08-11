"""Amazon Bedrock LLM client (Anthropic Messages API on Bedrock).

Pay-per-call only — no standing infrastructure cost, which is why the platform
leans on Bedrock rather than a hosted model endpoint. Credentials come from the
standard AWS chain (env / profile / instance role); none are read here.
"""

from __future__ import annotations

import json
from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

from legalintel.config import Settings
from legalintel.llm.base import LLMClient, LLMError, LLMTransientError

# Bedrock error codes worth retrying vs. failing fast.
_RETRYABLE = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException"}


class BedrockClient(LLMClient):
    def __init__(self, settings: Settings) -> None:
        import boto3  # imported lazily so `mock` provider needs no AWS deps

        self._settings = settings
        self._model_id = settings.bedrock_model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=Config(
                read_timeout=int(settings.llm_timeout_seconds) + 5,
                retries={"max_attempts": 1},  # our tenacity layer owns retries
            ),
        )

    def complete_json(self, *, system: str, prompt: str, timeout: float) -> str:
        # NOTE: sampling params (temperature/top_p/top_k) are intentionally omitted.
        # The current Claude models on Bedrock (Opus 4.8 / Sonnet 5) reject them
        # with a 400; older models accept them. Leaving them off works everywhere,
        # and grounding is enforced by the validation agent regardless of sampling.
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = self._client.invoke_model(modelId=self._model_id, body=json.dumps(body))
            payload = json.loads(resp["body"].read())
            return str(payload["content"][0]["text"])
        except ReadTimeoutError as exc:
            raise LLMTransientError(f"bedrock read timeout: {exc}") from exc
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _RETRYABLE:
                raise LLMTransientError(f"bedrock throttled/unavailable: {code}") from exc
            raise LLMError(f"bedrock call failed: {code}: {exc}") from exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMTransientError(f"unexpected bedrock payload: {exc}") from exc
