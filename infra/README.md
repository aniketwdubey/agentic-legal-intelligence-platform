# Infrastructure (AWS CDK)

A **lean, cheap, fully destroyable** data plane **+ serverless API**. It
provisions only free, near-free, or pure pay-per-call resources so it's safe to
spin up and tear down repeatedly on minimal credits.

## What gets created

| Resource | Purpose | Cost |
|---|---|---|
| S3 bucket | Holds the legal corpus (`corpus/*.json`) | ~cents/mo for a few KB |
| S3 BucketDeployment | Uploads `../data/corpus` at deploy time | free (runs once) |
| IAM role | Least-privilege: `bedrock:InvokeModel` + read-only S3 + Lambda logs | free |
| Lambda (container image) | Runs the FastAPI app; reads corpus from S3, calls Bedrock | per-request only |
| Lambda **Function URL** | Public HTTPS endpoint for the API (no API Gateway) | free |

**Not created** (the expensive stuff): OpenSearch, RDS, NAT gateways, always-on
compute. Bedrock is **pay-per-call** and Lambda bills only per request, so there
is no standing cost — an idle stack costs ~cents/month for the S3 objects.

The Lambda reuses the **same Docker image** the tests and `docker-compose` run.
The [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
is baked into the image (an execution-environment extension, dormant outside
Lambda), so uvicorn runs unchanged behind the Function URL — no handler shim, no
Mangum, and identical behaviour locally and in the cloud.

## Prerequisites

- AWS credentials with rights to create the above (`aws configure` / SSO).
- Node.js (for the CDK CLI via `npx aws-cdk`) and Python 3.11+.
- **Docker running** — CDK builds the Lambda container image at deploy time.
- One-time Bedrock model access approved in the console for the target region.

## Create / destroy

```bash
cd infra
make install          # venv + aws-cdk-lib
make bootstrap        # one-time per account/region
make deploy           # create everything  (prints ApiUrl, CorpusBucketName, ...)
# ... use it ...
make destroy          # remove everything; the bucket is auto-emptied first
```

Deploy targets `us-east-1` by default (broadest Bedrock model access); override
with `CDK_DEFAULT_REGION` or the standard `AWS_REGION`. Pick the model with
`make deploy -- -c model_id=us.anthropic.claude-sonnet-4-6` (any inference-profile
id your account exposes; see `scripts/check_bedrock.py`).

## Calling the API

The stack outputs `ApiUrl`. By default the Function URL is **`AWS_IAM`**-authed
so a leaked URL can't run up a Bedrock bill — call it with SigV4-signed requests:

```bash
API=$(aws cloudformation describe-stacks --stack-name LegalIntelStack \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)

# awscurl signs with your AWS credentials (pipx install awscurl)
awscurl --service lambda --region us-east-1 -X POST "${API}v1/query" \
  -H 'content-type: application/json' \
  -d '{"question":"What is the legal standard for a motion to dismiss?"}'
```

For a **browser-friendly public** endpoint (no signing), deploy with
`make deploy -- -c auth=none`. Only do this knowingly: the endpoint invokes
Bedrock on every call, so an unauthenticated URL is a cost-abuse vector.

## Run the app locally against the deployed data plane

```bash
export LEGALINTEL_CORPUS_S3_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name LegalIntelStack \
  --query "Stacks[0].Outputs[?OutputKey=='CorpusBucketName'].OutputValue" --output text)
export LEGALINTEL_LLM_PROVIDER=bedrock
make -C .. run
```
