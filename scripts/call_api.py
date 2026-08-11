"""Call the deployed Lambda Function URL with a SigV4-signed request.

The Function URL is AWS_IAM-authed, so a plain browser/curl gets 403 Forbidden;
every request must be signed with your AWS credentials. This script signs with
the standard AWS credential chain (``aws configure`` / SSO / env) — same creds
the CLI uses — so no secrets live here.

Usage:
    python scripts/call_api.py                        # GET /health + a demo query
    python scripts/call_api.py "your legal question"  # GET /health + your query

The endpoint is discovered from the CloudFormation stack output so it stays
correct across redeploys; override with LEGALINTEL_API_URL to point elsewhere.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = os.environ.get("LEGALINTEL_AWS_REGION", "us-east-1")
STACK = os.environ.get("LEGALINTEL_STACK_NAME", "LegalIntelStack")
# Function URLs sign against the "lambda" service (not "execute-api").
SERVICE = "lambda"


def _resolve_base_url() -> str:
    """Return the API base URL (no trailing slash), from env or the CFN output."""
    if override := os.environ.get("LEGALINTEL_API_URL"):
        return override.rstrip("/")
    cfn = boto3.client("cloudformation", region_name=REGION)
    outputs = cfn.describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
    for out in outputs:
        if out["OutputKey"] == "ApiUrl":
            return str(out["OutputValue"]).rstrip("/")
    raise SystemExit(f"ApiUrl output not found on stack {STACK}; is it deployed?")


def call(base: str, method: str, path: str, body: str | None) -> None:
    url = base + path
    data = body.encode() if body else None
    signable = AWSRequest(
        method=method,
        url=url,
        data=data,
        headers={"content-type": "application/json"} if body else {},
    )
    SigV4Auth(boto3.Session().get_credentials(), SERVICE, REGION).add_auth(signable)
    request = urllib.request.Request(
        url, data=data, method=method, headers=dict(signable.headers.items())
    )
    try:
        with urllib.request.urlopen(request, timeout=100) as resp:
            print(f"{method} {path} -> {resp.status}\n{resp.read().decode()}")
    except urllib.error.HTTPError as exc:
        print(f"{method} {path} -> {exc.status}\n{exc.read().decode()}")


def main(argv: list[str]) -> int:
    base = _resolve_base_url()
    call(base, "GET", "/health", None)
    print("-" * 60)
    question = argv[0] if argv else "What is the legal standard for a motion to dismiss?"
    call(base, "POST", "/v1/query", json.dumps({"question": question}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
