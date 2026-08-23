# Agentic Legal Intelligence Platform

An **intent-agnostic, multi-agent** legal assistant where a planner interprets a
request at runtime, retrieval fetches supporting authority, a drafting agent
composes an answer, and a **citation-verification agent** checks every legal
claim against the retrieved source **before anything reaches the user**. When it
cannot ground a claim, it **abstains or escalates** instead of inventing law.

> Hallucinated case law is a career-ending failure mode for legal AI (lawyers
> have been sanctioned for filing AI-invented citations). This platform's
> headline property is that **it cannot assert a legal proposition without a
> verified citation to retrieved source text.**

Built as an independent portfolio project on public legal data only, deployed on
the AWS agent stack — **Strands Agents SDK + Amazon Bedrock + AgentCore Runtime &
Memory** (multi-turn), all via CDK.

---

## Architecture

```
  QueryRequest(question)        conversation history (optional — supplied by
        │                        AgentCore Memory when deployed; see below)
        │                                         │
        ▼                                         ▼
  ┌──────────────────────────  SUPERVISOR  ──────────────────────────┐
  │   fixed pipeline · validates every hop · trace_id + confidence    │
  └───────────────────────────────────────────────────────────────────┘
        │
  ①  Planner    (Strands Agent · LLM)  →  interprets intent: task_type +
        │                                 search_queries  (uses history to
        ▼                                 resolve follow-up references)
  ②  Retrieval  (Strands @tool)        →  hybrid BM25 + dense, min-max fused
        │                                 over the corpus → RetrievedAuthority[]
        ▼
  ③  Drafting   (Strands Agent · LLM)  →  DraftAnswer: every claim grounded
        │                                 ONLY in the retrieved set (id + quote)
        ▼
  ④  Validation (RULE-BASED · no LLM)  →  per claim: authority_id in the
        │                                 retrieved set? quote authentic?
        ▼                                 proposition grounded above threshold?
  Grounding policy   (only verified claims survive)
     ├─ none verified            → ABSTAIN   ("insufficient authority")
     ├─ low confidence / risk    → ESCALATE  (human review)
     └─ verified + confident     → ANSWER    (answer + verified citations)
        │
        ▼
  QueryResponse (status, answer, citations, confidence, trace)
```

Each agent owns a typed input/output contract (its own `schema.py`), so a failure
is attributable to a **specific agent / step / validation rule** rather than one
opaque prompt chain.

### The agents

| Agent | Input → Output | LLM? | Responsibility |
|---|---|---|---|
| **Planner** (Strands `Agent`) | `QueryRequest` (+ history) → `Plan` | ✅ | Interprets intent at runtime — classifies `task_type` (research / drafting / doc_review), the jurisdiction, and one or more `search_queries`. When conversation history is supplied (multi-turn), it resolves follow-up references (*"the first of those factors"*) against it. Asserts no legal proposition itself. (The `retrieve → draft → validate` sequence is a **fixed** pipeline the supervisor always runs, not chosen per request — see the note below.) |
| **Retrieval** (Strands `@tool`) | `list[str]` → `RetrievedAuthority[]` | ❌ | Runs each query through hybrid search (BM25 + dense embeddings) over the corpus, min-max fuses the two legs, and returns the top-k authorities with scores. Decoupled — takes plain queries, not a `Plan`. Pure-Python, no search service. |
| **Drafting** (Strands `Agent`) | question + `RetrievedAuthority[]` → `DraftAnswer` | ✅ | Composes the answer as a set of `claims`, each carrying an `authority_id` from the retrieved set and a **verbatim quote** copied from that authority's text. Insufficient context → empty claim list (→ abstention). |
| **Validation** (`validate()`) | `DraftAnswer` + `RetrievedAuthority[]` → `ValidationReport` | ❌ **rule-based** | Independently verifies each claim: (1) `citation_exists` — the `authority_id` is in the retrieved set; (2) the quote is a real span of that authority; (3) the claim is grounded above `grounding_threshold`. Emits per-claim verdicts + support scores. |
| **Supervisor** | orchestrates all of the above | ❌ | Owns the control loop: validates every agent output against its schema at the boundary, applies the grounding policy, computes confidence, and binds a `trace_id` across the run. |

### Supervisor control & grounding policy

The pipeline is a **fixed, deterministic sequence** — `plan → retrieve → draft →
validate → policy` — run by the supervisor in plain Python, *not* a step list the
model chooses. That's a deliberate design choice: the citation-verification
guarantee depends on validating the draft against the **exact** authority set that
was retrieved, so the control flow can't be left to an LLM. The planner shapes
*what* is retrieved (`task_type`, `search_queries`); the supervisor owns *how* the
steps run. (`Plan.steps` is descriptive metadata today, not a control input.)

On every hop the supervisor:

- **Validates at the boundary.** Each agent output is parsed into its pydantic
  schema, so a malformed LLM response is caught *where it was produced*, not
  downstream as a corrupt final answer.
- **Runs LLM agents through Strands.** Planner and drafter are Strands `Agent`s
  invoked with a Pydantic `structured_output_model`, so the framework enforces the
  output schema (via a forced tool call) and owns model retries — a malformed
  completion never reaches the pipeline as untyped data.
- **Applies the grounding policy** to the validation report to choose the status:
  - only **verified** claims survive — any claim whose citation isn't in the
    retrieved set, or whose quote/grounding fails, is dropped;
  - nothing survives → **`ABSTAIN`** ("insufficient authority to answer");
  - verified but low-confidence / high-risk → **`ESCALATE`** (human review);
  - verified + confident → **`ANSWER`** — the answer plus *only* its verified citations.
- **Computes confidence** from the support scores of the surviving claims, so the
  number in the response reflects how strongly the quotes actually grounded
  against the corpus.

The upshot: **a hallucinated citation cannot reach the user by construction** — it
is removed before the answer is assembled, and if nothing is left the system
abstains rather than guesses.

### Deployed runtime — Amazon Bedrock AgentCore

The pipeline above is host-agnostic. It's deployed on **Amazon Bedrock AgentCore
Runtime** as an ARM64 container, with **AgentCore Memory** providing per-session
conversation history — which is what makes the multi-turn follow-ups work:

```
  scripts/invoke_agentcore.py   (SigV4 InvokeAgentRuntime · runtimeSessionId)
        │
        ▼
  Amazon Bedrock AgentCore Runtime  (ARM64 container · /ping + /invocations)
        │                                   ┌──────── AgentCore Memory ────────┐
        │   load last-K turns  ────────────►│  get_last_k_turns(session)        │
        ▼                                   │  create_event(session, turn)      │
  BedrockAgentCoreApp @entrypoint  ────────►└───────────────▲──────────────────┘
        │   (store the new turn) ───────────────────────────┘
        ▼
  Supervisor pipeline  (planner → retrieval → drafting → validation → policy)
     └─ Planner / Drafting ──► Amazon Bedrock (Claude Haiku 4.5, pay-per-call)
        │
        ▼
  QueryResponse (JSON — status, answer, verified citations, confidence)
```

The **same supervisor code** runs offline (mock `StubModel`), locally (`make run`
serves the FastAPI app for dev), and on AgentCore (`agentcore_app.py` wraps it in
the Runtime contract). Memory is optional — without `LEGALINTEL_MEMORY_ID` the
agent is a stateless single-shot. Provisioned entirely by CDK — see
[`infra/README.md`](infra/README.md).

> The earlier **Lambda Function URL** deployment (container Lambda + AWS Lambda Web
> Adapter, no conversation memory) is preserved on the
> [`lambda-function-url`](../../tree/lambda-function-url) branch.

### Key design decisions

- **Built on the [Strands Agents SDK](https://strandsagents.com).** The LLM
  agents (planner, drafter) are Strands `Agent`s with native `BedrockModel` and
  Pydantic structured output; retrieval is a Strands `@tool`; cross-cutting
  monitoring is a Strands hook (`LoggingHook`). The framework owns the tool-loop,
  retries, and schema enforcement, so there is no bespoke provider/guardrail layer.
- **Validation is rule-based, not another LLM.** Verifying a citation is a factual
  check against the retrieved set; we do not want a second hallucination-prone
  model deciding whether the first hallucinated. It stays a plain, deterministic
  function (a first-class pipeline gate — deliberately *not* an LLM and *not* a
  hook): it checks (1) the cited `authority_id` is in the retrieved set and (2) the
  quote is a real span of that authority and the proposition is covered above a
  grounding threshold.
- **Pluggable model provider.** `mock` (an offline `StubModel` — deterministic,
  zero-cost, the default and what CI uses) and `bedrock` (Amazon Bedrock,
  pay-per-call), selected in `models.py:build_model()`. The `StubModel` implements
  the Strands `Model` interface so tests exercise the *same* agent code path as
  live Bedrock, with no network.
- **Hybrid retrieval with no search service.** BM25 (lexical) + dense embeddings
  (a deterministic hashing embedder by default, Titan on Bedrock), fused per
  query. Pure-Python so it runs in CI and in the container without OpenSearch.
- **Deployed on AgentCore Runtime with earned Memory.** The agent runs on Amazon
  Bedrock **AgentCore Runtime**; **AgentCore Memory** supplies per-session history
  so follow-up questions resolve. Memory is a *real* capability the deployment
  uses — not a hollow keyword — and the whole thing is native CDK
  (`AWS::BedrockAgentCore::{Memory,Runtime,RuntimeEndpoint}`).
- **Cost-conscious AWS.** Bedrock and AgentCore Runtime are pay-per-call and scale
  to zero; the CDK stack adds only near-free storage (Memory + the ECR image), so
  it spins up and tears down cheaply. See [`infra/`](infra/README.md).

### Layout

A modular monolith: everything for one agent (its `agent`, `schema`, and `prompt`)
lives in that agent's folder.

```
src/legalintel/
  config.py        pydantic-settings (env-driven, no secrets in code)
  models.py        build_model(): Strands BedrockModel | offline StubModel
  logging.py       structlog config + per-run trace_id binding
  schemas.py       shared cross-cutting contracts (Authority, Query*, Status, TaskType)
  agents/
    supervisor.py  orchestration, grounding policy, confidence, trace
    hooks.py       LoggingHook (cross-cutting agent monitoring)
    _stub_model.py offline Strands model for tests/CI (no network)
    planner/       agent + Plan schema + prompt        (LLM)
    drafting/      agent + DraftAnswer schema + prompt  (LLM)
    retrieval/     retrieve @tool                       (rule-based)
    validation/    validate() + ValidationReport schema (rule-based)
  retrieval/       corpus loader, BM25, embeddings, hybrid fusion, text utils
  api/             FastAPI app for local dev (thin handlers; no business logic)
  eval/            golden-set format, metrics, runner
agentcore_app.py   AgentCore Runtime entrypoint (BedrockAgentCoreApp + Memory)
Dockerfile.agentcore   ARM64 image for AgentCore (/ping + /invocations on 8080)
data/corpus/       tiny public-legal fixture corpus (statutes, a case, playbook clauses)
eval/golden_set.jsonl   golden questions → reference answer → expected authorities
tests/             pytest; model fully offline (StubModel); fixture corpus
infra/             AWS CDK app (AgentCore Runtime + Memory; fully destroyable)
scripts/           ask.py (CLI), invoke_agentcore.py (deployed runtime), check_bedrock.py
```

---

## Setup

Requires Python 3.11+. [`uv`](https://docs.astral.sh/uv/) is preferred; the
Makefile falls back to `venv`/`pip` if `uv` is absent.

```bash
cp .env.example .env      # defaults to the offline mock provider — no AWS needed
make install
```

## Run the slice

```bash
make run          # FastAPI on http://localhost:8000  (mock provider by default)
# in another shell:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/query -H 'content-type: application/json' \
  -d '{"question":"What are the statutory factors for fair use under US copyright law?"}' | jq
```

Or from the CLI:

```bash
make demo
python scripts/ask.py "When are consequential damages too remote to recover for breach of contract?"
python scripts/ask.py "What is the blood alcohol limit for pilots in Japan?"   # → abstains
```

### Docker

```bash
make docker-up    # builds the image and serves on :8000 (mock provider, JSON logs)
```

## Run the eval

```bash
make eval
```

Sample output on the fixture corpus + golden set (mock provider):

| metric | value |
|---|---|
| retrieval recall@k | **1.000** |
| citation precision | **1.000** |
| **hallucinated-citation rate** | **0.000** |
| correct-abstention rate | **1.000** |
| answered / abstained / escalated | 3 / 1 / 3 |

`hallucinated_citation_rate` is the headline metric — the fraction of emitted
citations not present in the retrieved set. The grounding policy guarantees this
is 0 by construction: nothing unverified is ever emitted. Wire the optional
thresholds into CI later to gate regressions:

```bash
python -m legalintel.eval.runner eval/golden_set.jsonl \
  --fail-under-recall 0.8 --max-hallucination 0.0
```

## Test / lint / type-check

```bash
make test    # pytest — no live LLM calls; runs fully offline
make lint     # ruff (lint + format check) + mypy --strict
```

---

## Using Amazon Bedrock (live)

Swaps the mock brain for real Claude. **Pay-per-call, no standing cost.**

1. `aws configure` (or SSO) so credentials resolve from the standard AWS chain.
2. **Enable model access** (one-time, per region): AWS console → Bedrock → Model
   access → request the Claude model. Models are off by default — this is the #1
   gotcha.
3. Confirm what you can invoke and set the id:

   ```bash
   python scripts/check_bedrock.py     # lists on-demand models + inference profiles
   ```

4. Point the app at Bedrock (defaults: Haiku 4.5, us-east-1):

   ```bash
   export LEGALINTEL_LLM_PROVIDER=bedrock
   export LEGALINTEL_AWS_REGION=us-east-1
   export LEGALINTEL_BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
   # export LEGALINTEL_EMBEDDER=bedrock   # optional: Titan dense embeddings
   make run        # or: python scripts/ask.py "What are the fair use factors?"
   ```

Credentials are never read from source. Current Claude models are **not**
on-demand invokable by their bare id — use the `us.anthropic.…` inference-profile
id from `check_bedrock.py` (already the default). Cheapest is Haiku 4.5; step up
to `us.anthropic.claude-sonnet-5` or `us.anthropic.claude-opus-4-8` for more
capability.

## Provision / tear down AWS (CDK)

Native CDK provisions the whole AgentCore stack — Memory, a least-privilege
execution role, the ARM64 image, and the Runtime + Endpoint. Pay-per-call and
fully destroyable:

```bash
cd infra && make install && make bootstrap   # one-time per account/region (needs Docker)
make deploy     # AgentCore Memory + IAM role + image + Runtime + Endpoint
make destroy    # removes everything (Memory has RemovalPolicy.DESTROY)
```

Invoke the deployed runtime with an IAM-signed `InvokeAgentRuntime` call. The
script discovers the runtime ARN from the stack output and, with no argument, runs
a **two-turn memory demo** — the follow-up resolves only because AgentCore Memory
supplies the prior turn:

```bash
python scripts/invoke_agentcore.py                       # 2-turn multi-turn demo
python scripts/invoke_agentcore.py "your legal question"
```

Prerequisites: one-time Bedrock model access in the console, and Docker running
for the image build. IAM, runtime, and cost details:
[`infra/README.md`](infra/README.md).

---

## Data (public only)

The repo ships a tiny **illustrative fixture corpus** so everything runs
offline: a US statute (17 U.S.C. § 107), a federal rule (FRCP 12(b)(6)), a
public-domain case (Hadley v. Baxendale, 1854), and self-authored standard
playbook clauses. Expand it into a real corpus from **public sources only** —
CourtListener / Caselaw Access Project (US case law), govinfo / US Code
(statutes), and CUAD (labelled contract clauses) — via
`scripts/fetch_corpus.py` (stubbed with the schema and sources documented).

## Scope & status

**Live today** — a grounded, citation-verified workflow with the full engineering
scaffold (config, logging, guardrails, tests, eval, Docker, CDK), **deployed on the
AWS agent stack**:

- **Framework — Strands Agents SDK:** planner and drafter are Strands `Agent`s with
  native `BedrockModel` + structured output; retrieval is a `@tool`; monitoring is
  a Strands hook.
- **Model — Amazon Bedrock:** Claude Haiku 4.5, pay-per-call, verified end-to-end.
- **Runtime — Bedrock AgentCore:** deployed on **AgentCore Runtime** (ARM64
  container) with **AgentCore Memory** for multi-turn conversation — provisioned
  entirely by CDK (`AWS::BedrockAgentCore::{Memory,Runtime,RuntimeEndpoint}`).

An earlier **Lambda Function URL** deployment (container Lambda + AWS Lambda Web
Adapter, no memory) is preserved on the
[`lambda-function-url`](../../tree/lambda-function-url) branch.

Deliberately **not** yet built (additive): exposing tools over **MCP** / AgentCore
Gateway, OpenTelemetry export + dashboard, an LLM-as-judge answer eval, a CI
regression gate on the eval metrics, and a real (CourtListener / CAP / CUAD) corpus
in place of the fixture set.
