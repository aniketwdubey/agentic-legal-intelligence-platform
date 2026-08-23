# Infrastructure (AWS CDK)

Native CDK for the **Amazon Bedrock AgentCore** deployment. Everything is
declarative, reviewable, and **fully `cdk destroy`-able** — pay-per-call, no
always-on compute.

## What gets created

| Resource | Purpose | Cost |
|---|---|---|
| `AWS::BedrockAgentCore::Memory` | Short-term, per-session conversation history (multi-turn) | ~cents/mo storage |
| Docker image asset (ARM64) | The agent container (`Dockerfile.agentcore`), built + pushed by CDK | ~cents/mo in ECR |
| IAM role | Least-privilege execution role the runtime assumes | free |
| `AWS::BedrockAgentCore::Runtime` | Hosts the agent container (HTTP protocol, `/ping` + `/invocations`) | per-request only |
| `AWS::BedrockAgentCore::RuntimeEndpoint` | Makes the runtime invocable (`InvokeAgentRuntime`) | free |

**Not created** (the expensive stuff): OpenSearch, RDS, NAT gateways, always-on
compute. Bedrock and AgentCore Runtime are **pay-per-call** and scale to zero, so
an idle stack costs only ~cents/month (Memory storage + the ECR image).

The container is **self-contained** — the corpus is baked into the image, so there
is no S3 data plane. The same supervisor code runs offline (mock), locally
(`make run`), and here (`agentcore_app.py` wraps it in the AgentCore contract).

## The execution role (least privilege)

The runtime assumes an IAM role scoped to exactly what the agent needs:

- `bedrock:InvokeModel` **and** `bedrock:InvokeModelWithResponseStream` — Strands'
  `BedrockModel` uses the **ConverseStream** API, which requires the streaming
  action (the classic `InvokeModel` alone is not enough).
- `bedrock-agentcore:*Event*` / memory read actions, scoped to the Memory ARN — so
  the agent can store and fetch conversation turns.
- `bedrock-agentcore:GetWorkloadAccessToken*` — required for the runtime's identity.
- ECR pull (the image), CloudWatch Logs, and X-Ray/metrics.

The trust policy allows `bedrock-agentcore.amazonaws.com` with `aws:SourceAccount`
+ `aws:SourceArn` conditions (confused-deputy protection).

## Prerequisites

- AWS credentials with rights to create the above (`aws configure` / SSO).
- Node.js (for the CDK CLI via `npx aws-cdk`) and Python 3.11+.
- **Docker running** — CDK builds the ARM64 container image at deploy time.
- One-time Bedrock model access approved in the console for the target region.

## Create / destroy

```bash
cd infra
make install          # venv + aws-cdk-lib (>=2.263 for the AgentCore constructs)
make bootstrap        # one-time per account/region
make deploy           # Memory + role + image + Runtime + Endpoint (prints RuntimeArn, MemoryId)
# ... use it ...
make destroy          # removes everything (Memory has RemovalPolicy.DESTROY)
```

Deploy targets `us-east-1` by default (broadest Bedrock model access); override
with `CDK_DEFAULT_REGION`. Pick the model with
`make deploy -- -c model_id=us.anthropic.claude-sonnet-4-6` (any inference-profile
id your account exposes; see `scripts/check_bedrock.py`).

## Invoking the runtime

`InvokeAgentRuntime` is IAM-authed. The helper discovers the runtime ARN from the
stack output, signs with your credentials, and runs a two-turn memory demo:

```bash
python ../scripts/invoke_agentcore.py                       # 2-turn multi-turn demo
python ../scripts/invoke_agentcore.py "your legal question"
```

A `runtimeSessionId` (>= 33 chars) ties turns together; AgentCore Memory supplies
the prior turns to the planner so follow-up questions resolve.

## Local development

You don't need the deployed runtime to develop. Run the agent container locally
(same image), or the FastAPI dev server:

```bash
# the AgentCore image locally (mock provider — no AWS needed):
docker run --rm -p 8080:8080 -e LEGALINTEL_LLM_PROVIDER=mock \
  $(docker buildx build -q --platform linux/arm64 --load -f ../Dockerfile.agentcore ..)
curl localhost:8080/ping
curl -s -X POST localhost:8080/invocations -d '{"prompt":"What are the fair use factors?"}'

# or the FastAPI dev server:
make -C .. run     # http://localhost:8000
```
