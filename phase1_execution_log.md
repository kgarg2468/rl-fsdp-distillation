# Phase 1 Execution Log (Real Pipeline Stability)

## Scope
- Config: `config/real_canary.toml`
- Stability target: one clean real canary with full artifacts, no integrity failures, spend within cap.
- Attempt cap: 3 attempts before escalation (plus 2 fallback retries executed on April 9, 2026 PDT).

## Gate A Result
- Real `preflight`: PASS
- Real `dryrun`: PASS
- Projected total spend (canary): `$2.5215`

## Attempt Records

### Attempt 1
- Run dir: `runs/run-011`
- Outcome: FAILED
- Failure stage: `distill` (stalled after `teacher_ft`)
- Error class: `runtime_stall`
- Evidence:
  - `run_manifest` frozen at `stages_completed=["rl","teacher_ft"]`
  - No eval/report artifacts generated
- Action taken:
  - Classified as runtime/transient stall
  - Process terminated manually after prolonged no-progress state

### Attempt 2
- Run dir: `runs/run-012`
- Outcome: FAILED (integrity gate)
- Failure stage: `eval` integrity checks (execution itself completed)
- Error class: `integrity_failed`
- Exact message:
  - `teacher overlap below baseline (teacher=0.2400, baseline=0.9733)`
- Evidence:
  - `runs/run-012/artifacts/eval/eval_metrics.json` -> `integrity.passed=false`
  - Full stages completed and all required artifacts present
- Action taken:
  - Applied minimal canary eval tuning for next attempt:
    - `max_tokens_eval: 48 -> 16`
    - `eval_max_tokens_candidates: [48, 96] -> [16, 32]`

### Attempt 3
- Run dir: `runs/run-013`
- Outcome: FAILED
- Failure stage: `distill`
- Error class: `runtime_transient_capacity`
- Exact message:
  - `Model creation for meta-llama/Llama-3.2-1B is paused. Reason: Tinker backend is running short on capacity, please wait.`
- Evidence:
  - Repeated pause messages during run execution
  - `run_manifest` stuck after `teacher_ft`
- Action taken:
  - Classified as external capacity blocker
  - Process terminated after classification

### Attempt 4 (Plan fallback retry 1)
- Run dir: `runs/run-014`
- Outcome: FAILED
- Failure stage: `distill` (no progress after `teacher_ft`)
- Error class: `runtime_stall`
- Config deltas applied before run:
  - `distillation.teacher_prompt_template: "raw" -> "numeric_strict"`
  - `distillation.filter_profile: "moderate" -> "strict"`
  - `distillation.epochs: 2 -> 1`
- Evidence:
  - `runs/run-014/artifacts/audit/run_manifest.json` -> `stages_completed=["rl","teacher_ft"]`
  - Missing `eval/report` artifacts for gate checks
  - `runs/run-014/artifacts/ledger.json` -> `total_spend_usd=0.0022`
- Action taken:
  - Classified as repeated runtime no-progress in distill window
  - Process terminated manually after prolonged no-progress state

### Attempt 5 (Plan fallback retry 2)
- Run dir: `runs/run-015`
- Outcome: FAILED
- Failure stage: `distill` (no progress after `teacher_ft`)
- Error class: `runtime_stall`
- Evidence:
  - `runs/run-015/artifacts/audit/run_manifest.json` -> `stages_completed=["rl","teacher_ft"]`
  - Missing `eval/report` artifacts for gate checks
  - `runs/run-015/artifacts/ledger.json` -> `total_spend_usd=0.0022`
- Action taken:
  - Re-classified as persistent runtime/capacity blocker after second fallback retry
  - Stopped retries per plan limit (2 fallback attempts), re-escalation required

## Phase 1 Status
- Phase 1 completion criteria: **NOT MET**
- What passed:
  - Real env/config readiness
  - At least one full real run with complete artifacts (Attempt 2)
  - Spend under canary cap
  - Canary patch config applied and verified through `preflight`/`dryrun` in real mode
- What failed:
  - Integrity requirement (`no integrity failures`) not met
  - Stable clean canary not achieved after 5 total attempts (including 2 fallback retries)
