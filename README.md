# RL + FSDP + Distillation

Budget-capped internship showcase pipeline for:
- RL training with Tinker API concepts
- Axolotl + FSDP fine-tuning stage
- Teacher -> student distillation
- Evaluation report with quality, cost, and training stability

## Runtime Modes
- `mock` (default): deterministic local adapters for reproducible demo runs.
- `real`: scaffolded adapter hooks for real Tinker/AWS integrations (preflight enforces required setup).

## Budget Targets
- Target Tinker spend window: **$20-$30**
- Default target cap: **$25**
- Hard stop cap: **$30**

Default token caps in `config/default.toml` project to ~`$20.24` total using 8B pricing assumptions.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Fast Commands
```bash
# Validate setup and projected spend
make preflight

# Print projected stage/token costs without running stages
make dryrun

# Run all local quality gates (style checks + tests + smoke)
make verify

# Run full mock pipeline
make all
```

You can override mode/config/state directory:
```bash
make all MODE=mock CONFIG=config/default.toml STATE_DIR=/tmp/inference-run
python -m inference_projects.cli preflight --mode real --config config/default.toml --state-dir /tmp/inference-real
```

## 10-Minute Demo Script
1. `make verify`
2. `make preflight MODE=mock`
3. `make dryrun MODE=mock`
4. `make all MODE=mock STATE_DIR=/tmp/inference-demo`
5. Open `/tmp/inference-demo/artifacts/reports/eval_report.md`
6. Show cost/quality/stability sections and explain mock vs real mode.

## Handoff Checklist
### I do (already automated in this repo)
- Budget guardrails (stage caps + hard cap + target warning band).
- Deterministic mock pipeline with reproducible artifacts.
- Preflight checks for env/config/writable state dir.
- Real adapter scaffolds with fail-fast setup errors.
- Test suite + local verify script + golden report fixture.

### You do (account-level setup I cannot perform)
- Create and secure credentials for Tinker and AWS.
- Export required real-mode env vars:
  - `TINKER_API_KEY`
  - `TINKER_BASE_URL`
  - `AWS_PROFILE`
  - `AWS_DEFAULT_REGION`
- Approve cloud quotas/billing/licensing requirements.
- Replace scaffolded real adapters with concrete API/job integrations.

## Commands
- `make rl`
- `make fsdp`
- `make distill`
- `make eval`
- `make report`
- `make all`
- `make smoke`
- `make preflight`
- `make dryrun`
- `make verify`
- `make test`

## Output Artifacts
- `artifacts/checkpoints/teacher/best_checkpoint.json`
- `artifacts/checkpoints/student/best_checkpoint.json`
- `artifacts/eval/eval_metrics.json`
- `artifacts/reports/eval_report.md`
- `artifacts/ledger.json`

## Notes
Real adapters are intentionally scaffolded with `NotImplementedError` until you connect credentials/tooling. Mock mode remains the default to keep onboarding and demo execution stable.
