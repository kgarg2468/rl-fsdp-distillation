# RL + FSDP + Distillation

## Project Aim
Build a reproducible, budget-capped training-and-evaluation pipeline that models an RL -> FSDP fine-tuning -> distillation lifecycle and produces auditable artifacts (checkpoints, metrics, spend ledger, and report).

This repository is intentionally a scaffold + roadmap:
- It already implements deterministic local pipeline mechanics in `mock` mode.
- It defines clear integration seams for production services in `real` mode.

## Current Status (Implemented vs Planned)
| Implemented now | Planned next |
| --- | --- |
| Deterministic mock stage adapters for `rl`, `fsdp`, `distill`, `eval` | Real Tinker RL adapter implementation |
| Stage budget checks + global hard-cap enforcement | Real Axolotl/FSDP launcher integration |
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
rl -> fsdp -> distill -> eval -> report
 |      |        |        |       |
 |      |        |        |       +--> artifacts/reports/eval_report.md
 |      |        |        +----------> artifacts/eval/eval_metrics.json
 |      |        +-------------------> artifacts/checkpoints/student/best_checkpoint.json
 |      +----------------------------> artifacts/checkpoints/teacher/best_checkpoint.json
 +-----------------------------------> artifacts/ledger.json (updated at every stage)
```

Run modes:
- `mock` (default): deterministic local adapters, reproducible outputs for demos/tests.
- `real`: adapter interfaces exist, but stage implementations intentionally raise `NotImplementedError` until integrations are wired.

## Tech Stack
- Language/runtime: Python `>=3.11` (uses `tomllib` from stdlib).
- Core stdlib modules: `argparse`, `dataclasses`, `tomllib`, `json`, `pathlib`.
- Configuration + artifacts: TOML config (`config/default.toml`) and JSON artifacts.
- Testing + quality: Pytest, custom style checks (`src/inference_projects/style_checks.py`), local verifier (`scripts/verify_local.sh`).
- Execution UX: Makefile targets plus CLI entrypoint (`python -m inference_projects.cli ...`).
- Design patterns: runtime adapter pattern (`mock` vs `real`), schema validators, persistent ledger model.
- Integration targets (scaffolded): Tinker API, AWS environment, Axolotl/FSDP training backends.

## Methods, Algorithms, and Evaluation Logic
### Implemented mechanics
- Token-based cost projection per stage from configured token caps and rates.
- Budget guardrails:
  - Per-stage budget cap checks.
  - Global hard-cap enforcement across cumulative projected spend.
- Deterministic actual-token scaling factors per stage to simulate under-projection execution in mock mode.
- Quality/cost reporting math:
  - Baseline/teacher/student benchmark values.
  - Student retention vs teacher.
  - Teacher vs student inference cost per 1k tokens and savings percentage.
- Training stability reporting per stage (`rl`, `fsdp`, `distill`) including NaN event counters.

### Planned algorithms and integrations
- Production RL method selection and concrete training job orchestration.
- True FSDP backend execution (Axolotl launch + distributed runtime wiring).
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
- `fsdp`: `$5.46`
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
make fsdp MODE=mock
make distill MODE=mock
make eval MODE=mock
make report MODE=mock
```

Full pipeline:
```bash
make all MODE=mock STATE_DIR=/tmp/inference-demo
```

CLI equivalents (public interface):
```bash
python -m inference_projects.cli preflight --mode mock --config config/default.toml --state-dir /tmp/inference-demo
python -m inference_projects.cli dryrun --mode mock --config config/default.toml --state-dir /tmp/inference-demo
python -m inference_projects.cli all --mode mock --config config/default.toml --state-dir /tmp/inference-demo
```

Supported CLI commands:
- `rl`
- `fsdp`
- `distill`
- `eval`
- `report`
- `all`
- `smoke`
- `preflight`
- `dryrun`

## Artifacts and Schemas
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
- `runtime`: default mode, warning band, required env vars for real mode

## Real-Mode Integration Plan
Current `real` mode status:
- Preflight enforces required environment variables:
  - `TINKER_API_KEY`
  - `TINKER_BASE_URL`
  - `AWS_PROFILE`
  - `AWS_DEFAULT_REGION`
- Stage adapters in real mode are scaffold placeholders that raise `NotImplementedError`.

Integration sequence:
1. Implement `RealRLAdapter.run(...)` with Tinker RL job submission and result normalization.
2. Implement `RealFSDPAdapter.run(...)` with Axolotl/FSDP launch + checkpoint materialization.
3. Implement `RealDistillAdapter.run(...)` with teacher output generation + student training.
4. Implement `RealEvalAdapter.run(...)` with benchmark + LLM-judge data ingestion and scoring.
5. Preserve current schema outputs so existing report + ledger pipeline remains stable.

## Limitations and Non-Goals
- Not a production training platform in current form.
- No real distributed training execution is wired yet.
- Mock-stage metrics are deterministic fixtures, not live benchmark results.
- No cloud provisioning, secrets management, or infrastructure-as-code included.
- No claim of a shipped production RL algorithm implementation.

## Roadmap
1. Replace each real adapter scaffold with integrated service-backed execution.
2. Add integration tests for real mode with safe stubs/mocks for external dependencies.
3. Add richer artifact lineage and observability (run IDs, job links, trace metadata).
4. Parameterize evaluation suites and benchmark datasets.
5. Introduce reproducible experiment packaging for handoff across environments.

## Troubleshooting
- `ModuleNotFoundError: No module named 'tomllib'`:
  - Use Python `3.11+` (this project requires stdlib `tomllib`).
- `Preflight failed for mode 'real'` with missing env vars:
  - Export required real-mode variables before running `--mode real`.
- `Projected total ... exceeds hard cap` or stage cap exceeded:
  - Adjust `budget` and/or `token_caps` in `config/default.toml`.
- `Required artifact missing` during later stages:
  - Run stages in order (`rl` -> `fsdp` -> `distill` -> `eval` -> `report`) or use `all`.
- `State directory is not writable`:
  - Pass a writable `--state-dir` or `STATE_DIR`.
