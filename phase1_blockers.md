# Phase 1 Blockers (Escalation After 5 Attempts)

## Blocker 1: Distill-stage runtime stalls / capacity pause
- Type: Runtime / external dependency
- Impact: Prevents reliable completion of real canary in bounded time
- Primary evidence:
  - `runs/run-011` stalled at distill after `teacher_ft`
  - `runs/run-013` repeated capacity pause for model creation
  - `runs/run-014` stalled after `teacher_ft` during distill window
  - `runs/run-015` stalled after `teacher_ft` during distill window
- Owner: Tinker platform / infra owner
- Requested action:
  - Increase or prioritize capacity for `meta-llama/Llama-3.2-1B` creation in this project window
  - Provide expected wait/retry windows for capacity pause events
- ETA requested: same day if possible, else next business day

## Blocker 2: Integrity gate failure in completed real run
- Type: Model quality / integrity
- Impact: Even when run completes, Phase 1 cannot be marked complete
- Primary evidence:
  - `runs/run-012/artifacts/eval/eval_metrics.json`:
    - `integrity.passed=false`
    - reason: `teacher overlap below baseline (teacher=0.2400, baseline=0.9733)`
  - Attempts `run-014` and `run-015` did not reach `eval`, so integrity re-validation under patched config is still blocked by runtime stalls.
- Owner: Pipeline/modeling owner (this repo)
- Requested action:
  - Diagnose teacher_ft degradation path on canary prompts
  - Adjust teacher_ft/distill canary behavior to keep teacher overlap above baseline
- ETA target: 1-2 working sessions

## Proposed Immediate Next Actions
1. Re-run Phase 1 once Tinker confirms healthy capacity window for distill model creation.
2. Keep current canary patch for next run set:
   - `teacher_prompt_template="numeric_strict"`
   - `filter_profile="strict"`
   - `epochs=1`
   - `max_tokens_eval=16`
3. If runtime stabilizes, execute one fresh real canary and gate on:
   - full stage completion through `report`
   - `integrity.passed=true`
   - required artifacts present/readable
   - `ledger.total_spend_usd <= 2.5215`
