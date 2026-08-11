#!/usr/bin/env python3
"""CDK app entry point.

Deliberately minimal so it costs almost nothing and tears down cleanly:
    cd infra && cdk deploy      # create everything
    cd infra && cdk destroy     # remove everything (bucket auto-emptied)
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from legalintel_infra.stack import LegalIntelStack

app = cdk.App()

LegalIntelStack(
    app,
    "LegalIntelStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    description="Agentic Legal Intelligence Platform — lean, cheap, fully destroyable data plane.",
)

app.synth()
