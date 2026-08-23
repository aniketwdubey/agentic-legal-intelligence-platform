"""Invoke the deployed AgentCore runtime (InvokeAgentRuntime data plane).

    python scripts/invoke_agentcore.py "your legal question"
    python scripts/invoke_agentcore.py            # runs a 2-turn memory demo

The runtime ARN is discovered from the CloudFormation stack output, and requests
are IAM-signed by boto3. A stable ``runtimeSessionId`` (>=33 chars) ties turns
together so AgentCore Memory can supply prior context for follow-ups.
"""

from __future__ import annotations

import json
import sys
import uuid

import boto3

REGION = "us-east-1"
STACK = "LegalIntelStack"


def _runtime_arn() -> str:
    cfn = boto3.client("cloudformation", region_name=REGION)
    outs = cfn.describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
    for o in outs:
        if o["OutputKey"] == "RuntimeArn":
            return str(o["OutputValue"])
    raise SystemExit("RuntimeArn output not found; is the stack deployed?")


def ask(client, arn: str, session_id: str, question: str) -> dict:
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        qualifier="live",
        runtimeSessionId=session_id,
        contentType="application/json",
        accept="application/json",
        payload=json.dumps({"prompt": question}).encode(),
    )
    body = resp["response"].read()
    return dict(json.loads(body))


def _show(turn: str, q: str, out: dict) -> None:
    print(f"\n=== {turn}: {q}")
    print(f"status={out.get('status')} confidence={out.get('confidence')}")
    print("answer:", (out.get("answer") or out.get("abstention_reason") or "")[:600])


def main(argv: list[str]) -> int:
    client = boto3.client("bedrock-agentcore", region_name=REGION)
    arn = _runtime_arn()
    session = f"legalintel-session-{uuid.uuid4().hex}"  # >= 33 chars

    if argv:
        _show("Q", argv[0], ask(client, arn, session, argv[0]))
        return 0

    # Two-turn memory demo: the follow-up only resolves via conversation memory.
    q1 = "What are the four statutory factors for fair use under US copyright law?"
    q2 = "Explain the first of those factors in more detail."
    _show("Turn 1", q1, ask(client, arn, session, q1))
    _show("Turn 2 (follow-up, same session)", q2, ask(client, arn, session, q2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
