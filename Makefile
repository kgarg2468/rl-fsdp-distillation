PYTHON ?= python3
MODE ?= mock
CONFIG ?= config/default.toml
STATE_DIR ?= .

.PHONY: install test rl fsdp distill eval report all smoke preflight dryrun verify

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

rl:
	$(PYTHON) -m inference_projects.cli rl --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

fsdp:
	$(PYTHON) -m inference_projects.cli fsdp --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

distill:
	$(PYTHON) -m inference_projects.cli distill --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

eval:
	$(PYTHON) -m inference_projects.cli eval --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

report:
	$(PYTHON) -m inference_projects.cli report --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

all:
	$(PYTHON) -m inference_projects.cli all --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

smoke:
	$(PYTHON) -m inference_projects.cli smoke --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

preflight:
	$(PYTHON) -m inference_projects.cli preflight --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

dryrun:
	$(PYTHON) -m inference_projects.cli dryrun --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

verify:
	./scripts/verify_local.sh
