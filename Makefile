PYTHON ?= python3
MODE ?= mock
CONFIG ?= config/default.toml
STATE_DIR ?= .
PRIOR_LEDGER ?=
PROJECT_HARD_CAP_USD ?= 35.0
RUNS_ROOT ?= $(CURDIR)/runs

.PHONY: install test rl teacher_ft distill eval report all smoke preflight dryrun campaign run-dir all-run campaign-run verify

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

rl:
	$(PYTHON) -m inference_projects.cli rl --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

teacher_ft:
	$(PYTHON) -m inference_projects.cli teacher_ft --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR)

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

campaign:
	@if [ -n "$(PRIOR_LEDGER)" ]; then \
		$(PYTHON) -m inference_projects.cli campaign --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR) --prior-ledger $(PRIOR_LEDGER) --project-hard-cap-usd $(PROJECT_HARD_CAP_USD); \
	else \
		$(PYTHON) -m inference_projects.cli campaign --mode $(MODE) --config $(CONFIG) --state-dir $(STATE_DIR) --project-hard-cap-usd $(PROJECT_HARD_CAP_USD); \
	fi

run-dir:
	@$(PYTHON) scripts/allocate_run_dir.py --root "$(RUNS_ROOT)"

all-run:
	@STATE_DIR="$$($(PYTHON) scripts/allocate_run_dir.py --root "$(RUNS_ROOT)")"; \
	echo "Allocated run directory: $$STATE_DIR"; \
	$(PYTHON) -m inference_projects.cli all --mode $(MODE) --config $(CONFIG) --state-dir "$$STATE_DIR"

campaign-run:
	@STATE_DIR="$$($(PYTHON) scripts/allocate_run_dir.py --root "$(RUNS_ROOT)")"; \
	echo "Allocated run directory: $$STATE_DIR"; \
	PRIOR="$(PRIOR_LEDGER)"; \
	if [ -z "$$PRIOR" ]; then \
		PRIOR="$$($(PYTHON) scripts/allocate_run_dir.py --root "$(RUNS_ROOT)" --latest-ledger || true)"; \
	fi; \
	if [ "$(MODE)" = "real" ] && [ -z "$$PRIOR" ]; then \
		echo "No prior ledger found under $(RUNS_ROOT). Set PRIOR_LEDGER=... for real campaign runs."; \
		exit 2; \
	fi; \
	if [ -n "$$PRIOR" ]; then \
		echo "Using prior ledger: $$PRIOR"; \
		$(PYTHON) -m inference_projects.cli campaign --mode $(MODE) --config $(CONFIG) --state-dir "$$STATE_DIR" --prior-ledger "$$PRIOR" --project-hard-cap-usd $(PROJECT_HARD_CAP_USD); \
	else \
		$(PYTHON) -m inference_projects.cli campaign --mode $(MODE) --config $(CONFIG) --state-dir "$$STATE_DIR" --project-hard-cap-usd $(PROJECT_HARD_CAP_USD); \
	fi

verify:
	./scripts/verify_local.sh
