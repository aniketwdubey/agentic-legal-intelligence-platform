"""The one and only stack — a lean, cheap, fully destroyable data plane + API.

What it creates (all either free, a few cents/month, or pure pay-per-call):
  * an S3 bucket holding the legal corpus (auto-emptied on destroy);
  * the corpus JSON uploaded into it at deploy time;
  * a least-privilege IAM role the compute layer assumes (Bedrock invoke +
    read-only S3 + Lambda logs);
  * a container-image Lambda running the FastAPI app behind a **Function URL** —
    the cheapest always-off compute (no API Gateway, no idle cost).

What it deliberately does NOT create: OpenSearch, RDS, NAT gateways, or any
always-on compute — those are the expensive parts. Bedrock is pay-per-call and
Lambda bills only per request, so there is no standing cost.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from constructs import Construct

# Bedrock model id used by the API. Defaults to the Haiku
# 4.5 cross-region inference profile (current Claude models are not on-demand
# invokable by bare id). Override at deploy with `-c model_id=...`.
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class LegalIntelStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        model_id = self.node.try_get_context("model_id") or DEFAULT_MODEL_ID
        # Function URL auth: default to AWS_IAM so a leaked URL can't run up a
        # Bedrock bill. Deploy with `-c auth=none` for a browser-friendly public
        # demo endpoint (understand the abuse risk before you do).
        auth = (self.node.try_get_context("auth") or "iam").lower()

        # --- corpus bucket (auto-emptied + destroyed on `cdk destroy`) --------
        corpus_bucket = s3.Bucket(
            self,
            "CorpusBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Upload the local corpus into the bucket under corpus/ at deploy time.
        s3deploy.BucketDeployment(
            self,
            "CorpusDeployment",
            sources=[s3deploy.Source.asset("../data/corpus")],
            destination_bucket=corpus_bucket,
            destination_key_prefix="corpus",
            prune=True,
            retain_on_delete=False,
        )

        # --- least-privilege app role ----------------------------------------
        app_role = iam.Role(
            self,
            "AppRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            description="Role the legal-intel compute layer assumes.",
            # CloudWatch Logs for the Lambda; nothing broader.
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        corpus_bucket.grant_read(app_role)
        app_role.add_to_policy(
            iam.PolicyStatement(
                # Strands' BedrockModel uses the Converse/ConverseStream API, which is
                # authorized by InvokeModel *and* InvokeModelWithResponseStream (the
                # streaming action) — both are required.
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                # We invoke via a cross-region *inference profile* (us.anthropic.*),
                # which fans out to the underlying foundation model in whichever
                # region it routes to. That requires InvokeModel on BOTH the
                # profile ARN and the foundation-model ARNs across regions, so the
                # foundation-model resources are intentionally region-agnostic.
                resources=[
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )

        # --- serverless API: container-image Lambda + Function URL ------------
        # Reuses the very Docker image the tests/compose run, so local and cloud
        # behaviour are identical. The AWS Lambda Web Adapter (baked into the
        # image) fronts uvicorn; no handler shim, no Mangum.
        api_fn = lambda_.DockerImageFunction(
            self,
            "ApiFn",
            code=lambda_.DockerImageCode.from_image_asset(
                "..",
                file="Dockerfile",
                # Pin the build platform so the image (incl. the Lambda Web Adapter
                # binary and compiled wheels) matches the function architecture
                # regardless of the host that runs `cdk deploy` — an arm64 Mac
                # would otherwise produce an arm64 image that fails on x86_64.
                platform=ecr_assets.Platform.LINUX_ARM64,
            ),
            architecture=lambda_.Architecture.ARM_64,  # Graviton: cheaper, native to the Mac build
            role=app_role,
            memory_size=1024,  # headroom for numpy + corpus index build on cold start
            timeout=Duration.seconds(90),  # planner + drafter LLM hops, with retries
            environment={
                "LEGALINTEL_LLM_PROVIDER": "bedrock",
                "LEGALINTEL_AWS_REGION": self.region,
                "LEGALINTEL_BEDROCK_MODEL_ID": model_id,
                # Read the corpus from S3 (the deployed data plane). Blanking the
                # local dir makes load_corpus fall through to the bucket.
                "LEGALINTEL_CORPUS_DIR": "",
                "LEGALINTEL_CORPUS_S3_BUCKET": corpus_bucket.bucket_name,
            },
        )

        fn_url = api_fn.add_function_url(
            auth_type=(
                lambda_.FunctionUrlAuthType.NONE
                if auth == "none"
                else lambda_.FunctionUrlAuthType.AWS_IAM
            ),
        )

        # --- outputs ----------------------------------------------------------
        CfnOutput(self, "CorpusBucketName", value=corpus_bucket.bucket_name)
        CfnOutput(self, "AppRoleArn", value=app_role.role_arn)
        CfnOutput(self, "ModelId", value=model_id)
        CfnOutput(self, "ApiUrl", value=fn_url.url)
        CfnOutput(self, "ApiAuthType", value="NONE" if auth == "none" else "AWS_IAM")
