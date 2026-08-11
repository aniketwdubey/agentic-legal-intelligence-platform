# syntax=docker/dockerfile:1
# Multi-stage build using uv for fast, reproducible installs.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- builder: install deps into a venv ---------------------------------------
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python .

# --- runtime -----------------------------------------------------------------
FROM base AS runtime
# AWS Lambda Web Adapter: lets this ordinary uvicorn server run behind a Lambda
# Function URL with no app changes. It is an execution-environment extension, so
# it is dormant when the image runs anywhere but Lambda (plain `docker run` /
# compose ignore /opt/extensions) — one image serves local, Docker, and Lambda.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

# Run as non-root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY data ./data
COPY eval ./eval
ENV PATH="/opt/venv/bin:$PATH" \
    LEGALINTEL_LOG_FORMAT=json \
    LEGALINTEL_CORPUS_DIR=/app/data/corpus \
    # Where the adapter forwards Lambda invocations (matches the uvicorn port);
    # gate cold starts on the app being ready to serve.
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "legalintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
