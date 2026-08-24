"""The AgentCore stack — the legal-intel agent on Amazon Bedrock AgentCore Runtime.

Everything is declarative CDK so it is reproducible, reviewable, and cleanly
`cdk destroy`-able:

  * an **AgentCore Memory** resource (short-term, per-session conversation);
  * an ARM64 **container image** (Dockerfile.agentcore) built + pushed by CDK;
  * a least-privilege **execution role** the runtime assumes (Bedrock invoke +
    Memory + image pull + logs + workload identity);
  * an **AgentCore Runtime** (HTTP protocol) running that image;
  * a **Runtime Endpoint** that makes it invocable.

The container is self-contained (corpus baked in), so unlike the Lambda variant
there is no S3 data plane. Bedrock and AgentCore are pay-per-call, so there is no
standing compute cost.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from constructs import Construct

# Bedrock model id the agent invokes (Haiku 4.5 cross-region inference profile).
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class LegalIntelStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        model_id = self.node.try_get_context("model_id") or DEFAULT_MODEL_ID

        # --- short-term conversation memory ----------------------------------
        memory = agentcore.CfnMemory(
            self,
            "ConvMemory",
            name="legalintel_conversations",
            event_expiry_duration=30,  # days a session's raw turns are retained
            description="Short-term multi-turn conversation memory for the legal-intel agent",
        )
        memory.apply_removal_policy(RemovalPolicy.DESTROY)

        # --- Bedrock Guardrail (safety layer, complements citation checks) ---
        # Input/output safety that sits alongside the rule-based grounding gate:
        # prompt-injection detection + PII redaction. This is defense-in-depth —
        # Guardrails guard the door, validation guards the substance.
        guardrail = bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name="legalintel-guardrail",
            description="Prompt-injection + PII protection for the legal-intel agent.",
            blocked_input_messaging="This request was blocked by the platform safety guardrail.",
            blocked_outputs_messaging="This response was blocked by the platform safety guardrail.",
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    # PROMPT_ATTACK covers prompt injection / jailbreak attempts (input only).
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type="PROMPT_ATTACK", input_strength="HIGH", output_strength="NONE"
                    ),
                ]
            ),
            sensitive_information_policy_config=(
                bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                    pii_entities_config=[
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="EMAIL", action="ANONYMIZE"
                        ),
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="PHONE", action="ANONYMIZE"
                        ),
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="US_SOCIAL_SECURITY_NUMBER", action="ANONYMIZE"
                        ),
                        bedrock.CfnGuardrail.PiiEntityConfigProperty(
                            type="CREDIT_DEBIT_CARD_NUMBER", action="BLOCK"
                        ),
                    ]
                )
            ),
        )
        guardrail.apply_removal_policy(RemovalPolicy.DESTROY)
        guardrail_version = bedrock.CfnGuardrailVersion(
            self, "GuardrailVersion", guardrail_identifier=guardrail.attr_guardrail_id
        )

        # --- container image (ARM64, built + pushed by CDK) ------------------
        image = ecr_assets.DockerImageAsset(
            self,
            "AgentImage",
            directory="..",
            file="Dockerfile.agentcore",
            platform=ecr_assets.Platform.LINUX_ARM64,  # AgentCore runs on Graviton
        )

        # --- least-privilege execution role ----------------------------------
        role = iam.Role(
            self,
            "AgentRole",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:*"
                    },
                },
            ),
            description="Execution role for the legal-intel AgentCore runtime.",
        )
        image.repository.grant_pull(role)  # pull the container image from ECR

        role.add_to_policy(
            iam.PolicyStatement(
                sid="Bedrock",
                # Strands' BedrockModel uses ConverseStream -> needs both actions.
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Guardrail",
                actions=["bedrock:ApplyGuardrail"],
                resources=[guardrail.attr_guardrail_arn],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Memory",
                actions=[
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:ListSessions",
                    "bedrock-agentcore:ListActors",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                    "bedrock-agentcore:GetMemory",
                    "bedrock-agentcore:GetMemoryRecord",
                    "bedrock-agentcore:ListMemoryRecords",
                ],
                resources=[memory.attr_memory_arn, f"{memory.attr_memory_arn}/*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="WorkloadIdentity",
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}"
                    ":workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}"
                    ":workload-identity-directory/default/workload-identity/*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Logs",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                    "logs:DescribeLogGroups",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}"
                    ":log-group:/aws/bedrock-agentcore/runtimes/*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Telemetry",
                actions=[
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                    "cloudwatch:PutMetricData",
                ],
                resources=["*"],
            )
        )

        # --- AgentCore Runtime + Endpoint ------------------------------------
        runtime = agentcore.CfnRuntime(
            self,
            "Runtime",
            agent_runtime_name="legalintel_agent",
            agent_runtime_artifact=agentcore.CfnRuntime.AgentRuntimeArtifactProperty(
                container_configuration=agentcore.CfnRuntime.ContainerConfigurationProperty(
                    container_uri=image.image_uri
                )
            ),
            network_configuration=agentcore.CfnRuntime.NetworkConfigurationProperty(
                network_mode="PUBLIC"  # IAM-authed InvokeAgentRuntime; no VPC for this demo
            ),
            protocol_configuration="HTTP",
            role_arn=role.role_arn,
            environment_variables={
                "LEGALINTEL_LLM_PROVIDER": "bedrock",
                "LEGALINTEL_AWS_REGION": self.region,
                "LEGALINTEL_BEDROCK_MODEL_ID": model_id,
                "LEGALINTEL_MEMORY_ID": memory.attr_memory_id,
                # Bedrock Guardrail applied on every model call (safety layer).
                "LEGALINTEL_BEDROCK_GUARDRAIL_ID": guardrail.attr_guardrail_id,
                "LEGALINTEL_BEDROCK_GUARDRAIL_VERSION": guardrail_version.attr_version,
                # AgentCore Observability: OTEL auto-instrumentation → CloudWatch.
                "AGENT_OBSERVABILITY_ENABLED": "true",
                "OTEL_RESOURCE_ATTRIBUTES": "service.name=legalintel-agent",
            },
        )
        runtime.node.add_dependency(role)

        endpoint = agentcore.CfnRuntimeEndpoint(
            self,
            "Endpoint",
            agent_runtime_id=runtime.attr_agent_runtime_id,
            # Track the runtime's current version so each deploy actually rolls the
            # endpoint forward — otherwise it stays pinned to the version it was
            # first created with and keeps serving the old image.
            agent_runtime_version=runtime.attr_agent_runtime_version,
            name="live",
        )

        # --- outputs ----------------------------------------------------------
        CfnOutput(self, "RuntimeArn", value=runtime.attr_agent_runtime_arn)
        CfnOutput(self, "EndpointArn", value=endpoint.attr_agent_runtime_endpoint_arn)
        CfnOutput(self, "MemoryId", value=memory.attr_memory_id)
        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
        CfnOutput(self, "ModelId", value=model_id)
