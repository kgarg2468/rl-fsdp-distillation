# RL + Teacher FT + Distillation

## Project Aim
Build a reproducible, budget-capped training-and-evaluation pipeline that models an RL -> Teacher FT fine-tuning -> distillation lifecycle and produces auditable artifacts (checkpoints, metrics, spend ledger, and report).

This repository provides both a deterministic local path and a live Tinker-backed path:
- It already implements deterministic local pipeline mechanics in `mock` mode.
- It runs end-to-end against Tinker services in `real` mode.

## Current Status (Implemented vs Planned)
| Implemented now | Planned next |
| --- | --- |
| Deterministic mock stage adapters for `rl`, `teacher_ft`, `distill`, `eval` | Real Tinker RL adapter implementation |
| Stage budget checks + global hard-cap enforcement | Real Axolotl/Teacher FT launcher integration |
| Schema validation for checkpoints, eval payload, and ledger | Real distillation job/data wiring |
| Ledger accounting for projected vs actual cost/token usage | Real benchmark + LLM-judge integrations |
| Markdown eval report generation with quality/cost/stability sections | Production telemetry and service-specific observability |
| CLI + Makefile workflow (`preflight`, `dryrun`, stage commands, `all`) | End-to-end real-mode execution |
| Pytest suite, smoke coverage, and golden report fixture | Productionized deployment/runtime automation |

## Architecture & Data Flow
The pipeline is orchestrated by `src/inference_projects/pipeline.py` and routed through stage adapters selected by runtime mode.

```text
config/default.toml
        |
        v
   preflight checks  ---> fail fast on invalid setup/budget/env
        |
        v
rl -> teacher_ft -> distill -> eval -> report
 |      |        |        |       |
 |      |        |        |       +--> artifacts/reports/eval_report.md
 |      |        |        +----------> artifacts/eval/eval_metrics.json
 |      |        +-------------------> artifacts/checkpoints/student/best_checkpoint.json
 |      +----------------------------> artifacts/checkpoints/teacher/best_checkpoint.json
 +-----------------------------------> artifacts/ledger.json (updated at every stage)
```

Run modes:
- `mock` (default): deterministic local adapters, reproducible outputs for demos/tests.
- `real`: stage adapters execute against Tinker (training checkpoints + sampling) and emit provider-derived usage metadata for ledger accounting.

## Tech Stack
- Language/runtime: Python `>=3.11` (uses `tomllib` from stdlib).
- Core stdlib modules: `argparse`, `dataclasses`, `tomllib`, `json`, `pathlib`.
- Configuration + artifacts: TOML config (`config/default.toml`) and JSON artifacts.
- Testing + quality: Pytest, custom style checks (`src/inference_projects/style_checks.py`), local verifier (`scripts/verify_local.sh`).
- Execution UX: Makefile targets plus CLI entrypoint (`python -m inference_projects.cli ...`).
- Design patterns: runtime adapter pattern (`mock` vs `real`), schema validators, persistent ledger model.
- Integration targets: Tinker API for training and sampling workloads.

## Methods, Algorithms, and Evaluation Logic
### Implemented mechanics
- Token-based cost projection per stage from configured token caps and rates.
- Budget guardrails:
  - Per-stage budget cap checks.
  - Global hard-cap enforcement across cumulative projected spend.
- Deterministic actual-token scaling factors per stage in mock mode.
- Real-mode accounting from adapter-emitted usage tokens (prefill/sample/train), with cost computed from configured rates when provider cost is unavailable.
- Quality/cost reporting math:
  - Baseline/teacher/student benchmark values.
  - Student retention vs teacher.
  - Teacher vs student inference cost per 1k tokens and savings percentage.
- Training stability reporting per stage (`rl`, `teacher_ft`, `distill`) including NaN event counters.

### Planned algorithms and integrations
- Production RL method selection and concrete training job orchestration.
- True Teacher FT backend execution (Axolotl launch + distributed runtime wiring).
- Production distillation data generation and student training loop.
- External evaluation service integrations (benchmarks and LLM-judge pipelines).

Note: this repo does not currently claim a specific production RL algorithm implementation (e.g., PPO/GRPO/DPO) as shipped.

## Budget Model and Guardrails
Default budget config (from `config/default.toml`):
- Target cap: `$25.00`
- Hard cap: `$30.00`
- Warning band: `$20.00-$30.00`

Default projected stage costs from token caps and rates:
- `rl`: `$8.39`
- `teacher_ft`: `$5.46`
- `distill`: `$5.06`
- `eval`: `$1.33`
- Total projected: `$20.24`

Guardrail behavior:
- Preflight validates stage-level projections and cumulative hard-cap feasibility.
- Runtime stage execution writes ledger records with projected + actual token/cost fields.
- Runs abort early when setup or budget constraints are violated.

## Runbook (Preflight, Dryrun, Stage Commands, Full Pipeline)
Setup:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Preflight (setup + budget checks):
```bash
make preflight MODE=mock
```

Dryrun (projection summary only, no artifacts):
```bash
make dryrun MODE=mock
```

Stage-by-stage commands:
```bash
make rl MODE=mock
make teacher_ft MODE=mock
make distill MODE=mock
make eval MODE=mock
make report MODE=mock
```

Full pipeline:
```bash
make all-run MODE=mock
```

Campaign pipeline (multi-seed with global cap):
```bash
make campaign-run MODE=real CONFIG=config/default.toml PROJECT_HARD_CAP_USD=35.0
```

Print/create next run directory under `<repo>/runs`:
```bash
make run-dir
```

CLI equivalents (public interface):
```bash
RUN_DIR="$(python scripts/allocate_run_dir.py --root runs)"
python -m inference_projects.cli preflight --mode mock --config config/default.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli dryrun --mode mock --config config/default.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli all --mode mock --config config/default.toml --state-dir "$RUN_DIR"

RUN_DIR="$(python scripts/allocate_run_dir.py --root runs)"
PRIOR_LEDGER="$(python scripts/allocate_run_dir.py --root runs --latest-ledger)"
python -m inference_projects.cli campaign --mode real --config config/default.toml --state-dir "$RUN_DIR" --prior-ledger "$PRIOR_LEDGER" --project-hard-cap-usd 35.0
```

Supported CLI commands:
- `rl`
- `teacher_ft`
- `distill`
- `eval`
- `report`
- `all`
- `smoke`
- `preflight`
- `dryrun`
- `campaign`

## Artifacts and Schemas
Run directory convention:
- All new runs go under `runs/run-###`.
- Single run artifacts: `runs/run-###/artifacts/...`
- Campaign artifacts: `runs/run-###/campaign/...`

Artifact contract under `artifacts/`:
- `artifacts/checkpoints/teacher/best_checkpoint.json`
- `artifacts/checkpoints/student/best_checkpoint.json`
- `artifacts/eval/eval_metrics.json`
- `artifacts/reports/eval_report.md`
- `artifacts/ledger.json`

Schema contract:
- Schema version: `1.0`
- Teacher checkpoint validator: required model/stage/quality/stability fields.
- Student checkpoint validator: teacher/student quality + compression/stability fields.
- Eval metrics validator: nested quality/cost/training_stability payloads.
- Ledger validator: total spend, stage spend map, token totals, and stage records.

Config contract (`config/default.toml`):
- `project`: name, seed
- `budget`: target/hard caps + per-stage budgets
- `token_rates_per_million`: prefill/sample/train rates
- `token_caps`: stage token caps
- `models`: teacher/student/baseline names
- `runtime`: default mode, warning band, required env vars for real mode, and real-mode polling settings
- `evaluation`: prompt fixture path + prompt limit for real eval
- `campaign`: seed list, min/max runs, bootstrap reps, and early-stop threshold

## Real-Mode Runbook
Required environment variables for `--mode real`:
- `TINKER_API_KEY`
- `TINKER_BASE_URL`

Recommended first run is a canary profile:
- Config: `config/real_canary.toml`
- Execute stage-by-stage before `all`.

Canary sequence:
```bash
set -a && source .env && set +a
RUN_DIR="$(python scripts/allocate_run_dir.py --root runs)"
python -m inference_projects.cli preflight --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli dryrun --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli rl --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli teacher_ft --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli distill --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli eval --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
python -m inference_projects.cli report --mode real --config config/real_canary.toml --state-dir "$RUN_DIR"
```

Notes:
- Real runs use dynamic run/checkpoint IDs from Tinker responses.
- Preflight guardrails still use projected spend; ledger records actual spend from runtime usage.

### Real-Mode Audit Bundle (Detailed Forensics)
In addition to existing artifacts, a real staged run now emits a dedicated audit bundle:
- `artifacts/audit/run_manifest.json`
- `artifacts/audit/stage_rl.json`
- `artifacts/audit/stage_teacher_ft.json`
- `artifacts/audit/stage_distill.json`
- `artifacts/audit/stage_eval.json`
- `artifacts/audit/eval_rows.jsonl`
- `artifacts/reports/run_audit_report.md`

Each stage audit file includes:
- stage timing (`started_at`, `finished_at`, `duration_seconds`)
- projected vs actual spend and token usage
- cumulative spend snapshots before/after the stage
- provider lineage (`run_id`, `provider_raw`) and stage payload snapshot
- prompt-level traces when available (`_prompt_traces`)

`eval_rows.jsonl` contains full per-row eval evidence:
- prompt/reference text
- baseline/teacher/student outputs
- overlap scores and row-level win indicators

Recommended one-shot credited execution order (fresh state dir):
```bash
set -a && source .env && set +a
STATE_DIR="$(python scripts/allocate_run_dir.py --root runs)"
python -m inference_projects.cli preflight --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli dryrun --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli rl --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli teacher_ft --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli distill --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli eval --mode real --config config/default.toml --state-dir "$STATE_DIR"
python -m inference_projects.cli report --mode real --config config/default.toml --state-dir "$STATE_DIR"
```

Campaign orchestration (2-3 seeds with early stop + global cap):
```bash
set -a && source .env && set +a
STATE_DIR="$(python scripts/allocate_run_dir.py --root runs)"
PRIOR_LEDGER="$(python scripts/allocate_run_dir.py --root runs --latest-ledger)"
python -m inference_projects.cli campaign --mode real --config config/default.toml --state-dir "$STATE_DIR" --prior-ledger "$PRIOR_LEDGER" --project-hard-cap-usd 35.0
```

Campaign outputs:
- `campaign/frozen_prompts.jsonl`
- `campaign/campaign_summary.json`
- `campaign/campaign_report.md`
- `campaign/runs/seed-<seed>/...` (standard per-run artifacts)

## Limitations and Non-Goals
- Not a production training platform in current form.
- No real distributed training execution is wired yet.
- Mock-stage metrics are deterministic fixtures, not live benchmark results.
- No cloud provisioning, secrets management, or infrastructure-as-code included.
- No claim of a shipped production RL algorithm implementation.

## Roadmap
1. Improve stage-specific training logic depth (currently minimal canary-safe training/checkpoint flows).
2. Add richer artifact lineage and observability (job links, request IDs, timing traces).
3. Expand benchmark fixture coverage and scoring strategies.
4. Harden retry/backoff strategies for long-running real workloads.
5. Introduce reproducible experiment packaging for handoff across environments.

## Troubleshooting
- `ModuleNotFoundError: No module named 'tomllib'`:
  - Use Python `3.11+` (this project requires stdlib `tomllib`).
- `Preflight failed for mode 'real'` with missing env vars:
  - Export `TINKER_API_KEY` and `TINKER_BASE_URL` before running `--mode real`.
- `Projected total ... exceeds hard cap` or stage cap exceeded:
  - Adjust `budget` and/or `token_caps` in `config/default.toml`.
- `Required artifact missing` during later stages:
  - Run stages in order (`rl` -> `teacher_ft` -> `distill` -> `eval` -> `report`) or use `all`.
- `State directory is not writable`:
  - Pass a writable `--state-dir` or `STATE_DIR`.
