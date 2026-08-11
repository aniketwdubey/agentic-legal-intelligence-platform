"""Corpus loading.

Authorities are stored as JSON files (one document per file) under
``corpus_dir``. In deployed environments the CDK stack syncs the same JSON to an
S3 bucket and sets ``corpus_s3_bucket``; ``load_corpus`` prefers local files and
falls back to S3 so tests never need the network.
"""

from __future__ import annotations

from pathlib import Path

from legalintel.config import Settings
from legalintel.logging import get_logger
from legalintel.schemas import Authority

log = get_logger("corpus")


def load_corpus(settings: Settings) -> list[Authority]:
    """Load and validate all authority documents for the configured corpus."""
    local = _load_local(Path(settings.corpus_dir))
    if local:
        log.info("corpus.loaded", source="local", count=len(local), dir=settings.corpus_dir)
        return local
    if settings.corpus_s3_bucket:
        remote = _load_s3(settings)
        log.info("corpus.loaded", source="s3", count=len(remote), bucket=settings.corpus_s3_bucket)
        return remote
    log.warning("corpus.empty", dir=settings.corpus_dir)
    return []


def _load_local(directory: Path) -> list[Authority]:
    if not directory.is_dir():
        return []
    out: list[Authority] = []
    for path in sorted(directory.glob("*.json")):
        out.append(Authority.model_validate_json(path.read_text(encoding="utf-8")))
    return out


def _load_s3(settings: Settings) -> list[Authority]:
    import boto3

    s3 = boto3.client("s3", region_name=settings.aws_region)
    out: list[Authority] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.corpus_s3_bucket, Prefix="corpus/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = s3.get_object(Bucket=settings.corpus_s3_bucket, Key=key)["Body"].read()
            out.append(Authority.model_validate_json(body))
    return out
