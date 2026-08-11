.PHONY: install lint test run eval demo docker-build docker-up docker-down cdk-deploy cdk-destroy clean

# Prefer uv; fall back to the local venv if uv is not installed.
UV := $(shell command -v uv 2>/dev/null)
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:  ## Create env and install the package with dev + infra extras
ifdef UV
	uv venv $(VENV)
	uv pip install --python $(PY) -e ".[dev,infra]"
else
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,infra]"
endif

lint:  ## Ruff + mypy
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests
	$(VENV)/bin/mypy src

test:  ## Run the offline test suite (no live LLM calls)
	$(VENV)/bin/pytest

run:  ## Start the FastAPI app (mock provider by default; set LEGALINTEL_LLM_PROVIDER=bedrock for live)
	$(VENV)/bin/uvicorn legalintel.api.main:app --reload --port 8000

eval:  ## Run the eval harness over the golden set
	LEGALINTEL_LOG_LEVEL=WARNING $(PY) -m legalintel.eval.runner eval/golden_set.jsonl

demo:  ## Ask a single question from the CLI
	$(PY) scripts/ask.py "What are the statutory factors for fair use under US copyright law?"

docker-build:  ## Build the API image
	docker compose build

docker-up:  ## Run the API in Docker on :8000
	docker compose up

docker-down:
	docker compose down

cdk-deploy:  ## Provision the (cheap) AWS stack
	cd infra && $(MAKE) deploy

cdk-destroy:  ## Tear the AWS stack down completely
	cd infra && $(MAKE) destroy

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache **/__pycache__ infra/cdk.out
