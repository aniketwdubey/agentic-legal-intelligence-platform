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

Built as an independent portfolio project on public legal data only.

---

## Architecture

```
                 QueryRequest (question, jurisdiction?)
                              │
                              ▼
              ┌───────────  Supervisor  ───────────┐   validates every agent
              │        (orchestration + policy)     │   output at the boundary,
              │                                      │   records an execution trace
   ┌──────────┴─────────┐                            │
   ▼                    ▼                            ▼
Planner ──► Plan    Retrieval ──► RetrievedAuthority[]   Drafting ──► DraftAnswer
(intent,            (hybrid BM25 + dense,                (claims grounded ONLY in
 queries,            RRF-style fusion over               retrieved authority)
 steps)              the legal corpus)                        │
                                                              ▼
                                             Validation (RULE-BASED, not an LLM)
                                             ├─ citation_exists? (in retrieved set)
                                             └─ supported?       (quote authentic +
                                                                  claim grounded)
                              │
                              ▼
            Grounding policy:  verified claims only
              ├─ none verified            → ABSTAIN ("insufficient authority")
              ├─ low confidence / risk    → ESCALATE (human review)
              └─ verified + confident      → ANSWER (answer + verified citations)
                              │
                              ▼
             QueryResponse (status, answer, citations, confidence, trace)
```

Each agent has a typed input/output contract (`src/legalintel/schemas.py`), so a
failure is attributable to a **specific agent / step / validation rule** rather
than one opaque prompt chain.

### Key design decisions

- **Validation is rule-based, not another LLM.** Verifying a citation is a
  factual check against the retrieved set; we do not want a second
  hallucination-prone model deciding whether the first hallucinated. The
  validation agent independently checks (1) the cited `authority_id` is in the
  retrieved set and (2) the claim's quote is a real span of that authority and
  the proposition is covered above a grounding threshold.
- **Pluggable LLM provider.** `mock` (deterministic, offline, zero-cost — the
  default and what CI uses) and `bedrock` (Amazon Bedrock, pay-per-call). The
  guardrails — timeout, retry-with-backoff, schema validation — live in
  `llm/base.py:structured()`, not in any provider.
- **Hybrid retrieval with no search service.** BM25 (lexical) + dense embeddings
  (a deterministic hashing embedder by default, Titan on Bedrock), fused per
  query. Pure-Python so it runs in CI and on Lambda without OpenSearch.
- **Cost-conscious AWS.** Bedrock is pay-per-call; the CDK stack provisions only
  free/near-free resources (S3 + SSM + IAM) so it spins up and tears down
  cheaply. See [`infra/`](infra/README.md).

### Layout

```
src/legalintel/
  config.py        pydantic-settings (env-driven, no secrets in code)
  logging.py       structlog structured logging (+ per-run trace_id binding)
  schemas.py       typed contracts at every agent boundary
  llm/             provider protocol + guarded structured() + bedrock/mock/prompts
  retrieval/       corpus loader, BM25, embeddings, hybrid fusion, shared text utils
  agents/          planner, retrieval, drafting, validation, supervisor
  api/             FastAPI app (thin handlers; no business logic in routes)
  eval/            golden-set format, metrics, runner
data/corpus/       tiny public-legal fixture corpus (statutes, a case, playbook clauses)
eval/golden_set.jsonl   golden questions → reference answer → expected authorities
tests/             pytest; LLM fully mocked; offline fixture corpus
infra/             AWS CDK app (cheap, fully destroyable)
scripts/           ask.py (CLI), check_bedrock.py, fetch_corpus.py (stub)
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

```bash
cd infra && make install && make bootstrap   # one-time
make deploy     # S3 corpus bucket + SSM config + least-privilege IAM role
make destroy    # removes everything; the bucket is auto-emptied first
```

Details and the (optional) serverless-API path: [`infra/README.md`](infra/README.md).

---

## Data (public only)

The repo ships a tiny **illustrative fixture corpus** so everything runs
offline: a US statute (17 U.S.C. § 107), a federal rule (FRCP 12(b)(6)), a
public-domain case (Hadley v. Baxendale, 1854), and self-authored standard
playbook clauses. Expand it into a real corpus from **public sources only** —
CourtListener / Caselaw Access Project (US case law), govinfo / US Code
(statutes), and CUAD (labelled contract clauses) — via
`scripts/fetch_corpus.py` (stubbed with the schema and sources documented).

## Scope of this slice

This implements build-order steps 1–4 from the design doc: a grounded,
citation-verified single workflow with the full engineering scaffold (config,
logging, guardrails, tests, eval, Docker, CDK). Deliberately **not** yet built:
the Strands/AgentCore runtime swap, Lambda/SQS/SNS tool integrations, OpenTelemetry
export + dashboard, and the LLM-as-judge answer eval. The orchestration is a thin
custom supervisor with typed agent boundaries so those are additive, not rewrites.
