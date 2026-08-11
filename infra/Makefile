.PHONY: install bootstrap synth diff deploy destroy

VENV := .venv
# The CDK CLI ships via npm. `npx aws-cdk` needs no global install; override with
# CDK=cdk if you ran `npm i -g aws-cdk`. The Python lib comes from requirements.txt.
CDK ?= npx aws-cdk

install:  ## Create an isolated venv with the CDK Python libs
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

bootstrap:  ## One-time per account/region
	$(VENV)/bin/python -m ensurepip >/dev/null 2>&1 || true
	. $(VENV)/bin/activate && $(CDK) bootstrap

synth:
	. $(VENV)/bin/activate && $(CDK) synth

diff:
	. $(VENV)/bin/activate && $(CDK) diff

deploy:  ## Create/update the stack
	. $(VENV)/bin/activate && $(CDK) deploy --require-approval never

destroy:  ## Remove everything (S3 bucket is auto-emptied)
	. $(VENV)/bin/activate && $(CDK) destroy --force
