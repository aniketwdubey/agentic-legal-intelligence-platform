"""Show which Claude models your AWS account/region can actually invoke.

Bedrock model-id availability varies by account and region. Newer Claude models
are frequently only invokable through a **cross-region inference profile** whose
id carries a `us.` / `apac.` / `global.` prefix (e.g. us.anthropic.claude-haiku-4-5)
rather than the bare on-demand id. This script lists both so you can set the id
that works in LEGALINTEL_BEDROCK_MODEL_ID.

    python scripts/check_bedrock.py

Prereqs: `aws configure` done, and model access enabled in the Bedrock console
(Bedrock -> Model access -> request the Claude model).
"""

from __future__ import annotations

import boto3

from legalintel.config import get_settings


def main() -> int:
    settings = get_settings()
    region = settings.aws_region
    print(f"Region: {region}\n")

    bedrock = boto3.client("bedrock", region_name=region)

    print("== Foundation models (Anthropic) ==")
    resp = bedrock.list_foundation_models(byProvider="Anthropic")
    for m in resp.get("modelSummaries", []):
        lifecycle = m.get("modelLifecycle", {}).get("status", "-")
        on_demand = "ON_DEMAND" in m.get("inferenceTypesSupported", [])
        print(f"  {m['modelId']:55s} on_demand={on_demand} status={lifecycle}")

    print("\n== Cross-region inference profiles (use these ids if on-demand fails) ==")
    try:
        profiles = bedrock.list_inference_profiles().get("inferenceProfileSummaries", [])
        for p in profiles:
            if "anthropic" in p.get("inferenceProfileId", "").lower():
                print(f"  {p['inferenceProfileId']:55s} {p.get('status', '-')}")
        if not profiles:
            print("  (none returned)")
    except Exception as exc:  # noqa: BLE001 - informational tool
        print(f"  could not list inference profiles: {exc}")

    print(
        "\nSet LEGALINTEL_BEDROCK_MODEL_ID to a model with on_demand=True, or to a "
        "profile id above.\nThen: LEGALINTEL_LLM_PROVIDER=bedrock make run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
