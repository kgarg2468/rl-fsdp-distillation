# Real-Run CEO Showcase Target Plan

This document is the execution reference for getting the project to a CEO-ready showcase level focused on real Tinker runs and model growth.

## Objective
By showcase time, we should be able to prove:
1. Real Tinker runs complete end-to-end (`rl -> teacher_ft -> distill -> eval -> report`).
2. Model growth is measurable (student improves vs baseline, with strong retention vs teacher).
3. Results are repeatable (at least 2 seeds with consistent direction).
4. Cost is controlled (within hard caps, with full ledger/audit evidence).

## Phase 1: Real Pipeline Stability

### Goal
Establish a reliable real-mode execution path without integrity or runtime blockers.

### Steps
1. Run `preflight` and `dryrun` in real mode.
2. Execute one small real canary run with strict budget caps.
3. Capture and fix blocker failures:
- Missing required usage fields
- Integrity/schema validation failures
- Stage execution failures
4. Re-run the canary until end-to-end completion is stable.

### Exit Criteria
1. One real canary run completes all stages.
2. No integrity check failures in generated artifacts.
3. Spend remains under configured hard cap.

## Phase 2: First Full Real End-to-End Evidence

### Goal
Generate a full real-mode run that produces clean, presentation-ready evidence.

### Steps
1. Execute one full real pipeline run (`all` flow or equivalent stage sequence).
2. Verify output artifacts are complete and internally consistent.
3. Record key metrics:
- Baseline/teacher/student benchmark scores
- Student vs baseline and student vs teacher comparisons
- Total spend and per-stage spend
- Training stability signals
4. Save this run as the primary reference run for technical walkthrough.

### Exit Criteria
1. Full real run completes from `rl` through `report`.
2. Artifacts are complete and valid:
- `artifacts/eval/eval_metrics.json`
- `artifacts/ledger.json`
- `artifacts/reports/eval_report.md`
- `artifacts/reports/run_audit_report.md`
3. Metrics can be explained coherently in one pass.

## Phase 3: Growth Optimization Pass

### Goal
Demonstrate measurable model improvement from tuning/distillation decisions.

### Steps
1. Run targeted tuning focused on current failure points (`needs_debug` causes).
2. Evaluate candidate teacher/distill configurations against acceptance gates.
3. Promote only candidates that pass integrity and quality constraints.
4. Re-run real evaluation on promoted candidate(s).
5. Document before-vs-after growth metrics from baseline reference to tuned result.

### Exit Criteria
1. At least one candidate passes integrity and promotion gates.
2. Student shows positive movement vs baseline (non-zero uplift).
3. Teacher/student relationship is directionally sensible and explainable.

## Phase 4: Reproducibility and Confidence

### Goal
Show growth signal is not a one-off by validating across multiple seeds.

### Steps
1. Run a 2-seed real mini-campaign with frozen prompts.
2. Compare seed-level outcomes for direction and variance.
3. Validate spend profile consistency across seeds.
4. Confirm audit trail exists for each run.

### Exit Criteria
1. At least 2 real seeds complete successfully.
2. Improvement trend direction is consistent across seeds.
3. No run violates hard spend guardrails.

## Phase 5: Showcase Packaging

### Goal
Convert technical evidence into a clear CEO-ready narrative.

### Steps
1. Build the showcase story in this order:
- Problem and objective
- System architecture and guardrails
- Real-run evidence
- Growth outcomes
- Cost discipline
- Next milestones
2. Prepare one concise known-limitations-and-mitigation section.
3. Create a single-source summary table of all key run metrics.
4. Select final artifact paths to reference live during the walkthrough.

### Exit Criteria
1. End-to-end story is clear in 10-15 minutes.
2. Every claim in the story maps to an artifact.
3. Known gaps are transparent with actionable next steps.

## Phase 6: Final Readiness and Rehearsal

### Goal
Lock a stable demo package and ensure delivery confidence.

### Steps
1. Run one final rehearsal with timer.
2. Validate all referenced artifact files and paths are accessible.
3. Prepare answers to expected CEO questions:
- Why this proves real progress
- How much confidence we have in growth
- Cost/quality tradeoffs
- What is needed for production confidence
4. Freeze the final demo narrative and evidence set.

### Exit Criteria
1. Rehearsal completes within target time.
2. Q&A responses are evidence-backed.
3. Final package is stable and ready to present.

## Global Hard Gates (Must Pass Before Showcase)
1. At least one full real run completed successfully.
2. At least one clear growth signal vs baseline.
3. At least two real seeds with consistent trend direction.
4. Full artifact trail present for showcased results.
5. Spend remained within configured hard caps.

## Progress Tracker

Use this checklist to track completion:
- [ ] Phase 1 complete
- [ ] Phase 2 complete
- [ ] Phase 3 complete
- [ ] Phase 4 complete
- [ ] Phase 5 complete
- [ ] Phase 6 complete
- [ ] All global hard gates passed
