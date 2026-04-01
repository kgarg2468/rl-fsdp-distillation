from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any

from inference_projects import audit
from inference_projects import budget
from inference_projects import campaign as campaign_utils
from inference_projects.adapters import (
    DistillStageAdapter,
    EvalStageAdapter,
    FSDPStageAdapter,
    RLStageAdapter,
    select_stage_adapters,
)
from inference_projects.config import ProjectConfig, REQUIRED_STAGES, load_config
from inference_projects.ledger import Ledger, StageRecord, add_record, load_ledger, save_ledger
from inference_projects.preflight import PreflightResult, ensure_preflight_ready, run_preflight
from inference_projects.pricing import TokenUsage, cost_usd
from inference_projects.schemas import (
    SCHEMA_VERSION,
    validate_eval_metrics,
    validate_student_checkpoint,
    validate_teacher_checkpoint,
)
from inference_projects.tinker_runtime import REAL_USAGE_KEY
from inference_projects.tinker_runtime import EVAL_ROWS_KEY, PROMPT_TRACES_KEY

SUPPORTED_COMMANDS = {"rl", "fsdp", "distill", "eval", "report", "all", "smoke", "preflight", "dryrun", "campaign"}


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def ledger(self) -> Path:
        return self.artifacts / "ledger.json"

    @property
    def teacher_ckpt(self) -> Path:
        return self.artifacts / "checkpoints/teacher/best_checkpoint.json"

    @property
    def student_ckpt(self) -> Path:
        return self.artifacts / "checkpoints/student/best_checkpoint.json"

    @property
    def eval_metrics(self) -> Path:
        return self.artifacts / "eval/eval_metrics.json"

    @property
    def report_md(self) -> Path:
        return self.artifacts / "reports/eval_report.md"

    @property
    def audit_dir(self) -> Path:
        return self.artifacts / "audit"

    @property
    def run_manifest(self) -> Path:
        return self.audit_dir / "run_manifest.json"

    def stage_audit(self, stage: str) -> Path:
        return self.audit_dir / f"stage_{stage}.json"

    @property
    def eval_rows(self) -> Path:
        return self.audit_dir / "eval_rows.jsonl"

    @property
    def audit_report_md(self) -> Path:
        return self.artifacts / "reports/run_audit_report.md"


def run_pipeline_command(
    command: str,
    *,
    mode: str | None = None,
    config_path: Path | str = Path("config/default.toml"),
    state_dir: Path | str = Path("."),
    prior_ledger: Path | str | None = None,
    project_hard_cap_usd: float = 35.0,
) -> dict[str, object] | None:
    if command not in SUPPORTED_COMMANDS:
        raise ValueError(f"Unknown command: {command}")

    cfg = load_config(config_path)
    resolved_mode = mode or cfg.runtime.default_mode
    paths = PipelinePaths(Path(state_dir))

    if command == "preflight":
        result = run_preflight(mode=resolved_mode, cfg=cfg, state_dir=paths.root)
        return result.as_dict()

    if command == "dryrun":
        return _dryrun_summary(cfg, resolved_mode)

    if command == "campaign":
        return run_campaign(
            cfg=cfg,
            mode=resolved_mode,
            state_dir=paths.root,
            prior_ledger=Path(prior_ledger) if prior_ledger else None,
            project_hard_cap_usd=float(project_hard_cap_usd),
        )

    preflight_result = ensure_preflight_ready(mode=resolved_mode, cfg=cfg, state_dir=paths.root)
    adapters = select_stage_adapters(resolved_mode)

    if command == "rl":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode, preflight=preflight_result)
    elif command == "fsdp":
        run_fsdp(cfg, paths, adapter=adapters.fsdp, mode=resolved_mode, preflight=preflight_result)
    elif command == "distill":
        run_distill(cfg, paths, adapter=adapters.distill, mode=resolved_mode, preflight=preflight_result)
    elif command == "eval":
        run_eval(cfg, paths, adapter=adapters.eval, mode=resolved_mode, preflight=preflight_result)
    elif command == "report":
        run_report(cfg, paths, mode=resolved_mode, preflight=preflight_result)
    elif command == "all":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode, preflight=preflight_result)
        run_fsdp(cfg, paths, adapter=adapters.fsdp, mode=resolved_mode, preflight=preflight_result)
        run_distill(cfg, paths, adapter=adapters.distill, mode=resolved_mode, preflight=preflight_result)
        run_eval(cfg, paths, adapter=adapters.eval, mode=resolved_mode, preflight=preflight_result)
        run_report(cfg, paths, mode=resolved_mode, preflight=preflight_result)
    elif command == "smoke":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode, preflight=preflight_result)
    return None


def _dryrun_summary(cfg: ProjectConfig, mode: str) -> dict[str, object]:
    preflight = run_preflight(mode=mode, cfg=cfg, check_state_dir=False)
    stages: dict[str, dict[str, object]] = {}
    running = 0.0
    for stage in REQUIRED_STAGES:
        projected_cost = budget.projected_stage_cost_usd(stage, cfg)
        stage_cap = cfg.budget.stage_budgets_usd[stage]
        running += projected_cost
        stages[stage] = {
            "projected_cost_usd": projected_cost,
            "stage_cap_usd": stage_cap,
            "within_stage_cap": projected_cost <= stage_cap,
            "cumulative_projected_total_usd": round(running, 4),
        }

    return {
        "mode": mode,
        "projected_total_usd": budget.projected_total_cost_usd(cfg),
        "target_warning_band_usd": {
            "min": cfg.runtime.projection_warning_min_usd,
            "max": cfg.runtime.projection_warning_max_usd,
        },
        "stages": stages,
        "warnings": preflight.warnings,
    }


def _projected_and_actual_mock(stage: str, cfg: ProjectConfig) -> tuple[TokenUsage, float, TokenUsage, float]:
    projected_tokens = budget.stage_token_usage(stage, cfg)
    projected_cost = budget.projected_stage_cost_usd(stage, cfg)
    # Keep actual spend under projection for predictable low-cost demo behavior.
    actual_factor = {"rl": 0.94, "fsdp": 0.96, "distill": 0.93, "eval": 0.90}[stage]
    actual_tokens = TokenUsage(
        prefill=int(projected_tokens.prefill * actual_factor),
        sample=int(projected_tokens.sample * actual_factor),
        train=int(projected_tokens.train * actual_factor),
    )
    rates = cfg.token_rates_per_million
    actual_cost = cost_usd(
        actual_tokens,
        prefill_rate=rates["prefill"],
        sample_rate=rates["sample"],
        train_rate=rates["train"],
    )
    return projected_tokens, projected_cost, actual_tokens, actual_cost


def _stage_budget_check(stage: str, cfg: ProjectConfig, paths: PipelinePaths) -> tuple[Ledger, TokenUsage, float]:
    ledger = load_ledger(paths.ledger)
    projected_tokens = budget.stage_token_usage(stage, cfg)
    projected_cost = budget.projected_stage_cost_usd(stage, cfg)
    budget.ensure_within_stage_budget(stage, projected_cost, cfg)
    budget.ensure_within_hard_cap(current_total=ledger.total_spend_usd, incoming_cost=projected_cost, cfg=cfg)
    return ledger, projected_tokens, projected_cost


def _real_usage_from_payload(payload: dict[str, object], cfg: ProjectConfig) -> tuple[TokenUsage, float, dict[str, object]]:
    raw = payload.get(REAL_USAGE_KEY)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Real mode adapter payload missing required usage object '{REAL_USAGE_KEY}'")

    required = ("prefill_tokens", "sample_tokens", "train_tokens")
    for key in required:
        if key not in raw:
            raise RuntimeError(f"Real mode usage object missing required field: {key}")
        if not isinstance(raw[key], int):
            raise RuntimeError(f"Real mode usage field '{key}' must be int")

    tokens = TokenUsage(
        prefill=int(raw["prefill_tokens"]),
        sample=int(raw["sample_tokens"]),
        train=int(raw["train_tokens"]),
    )
    raw_cost = raw.get("cost_usd")
    if raw_cost is None:
        rates = cfg.token_rates_per_million
        actual_cost = cost_usd(
            tokens,
            prefill_rate=rates["prefill"],
            sample_rate=rates["sample"],
            train_rate=rates["train"],
        )
    elif isinstance(raw_cost, (int, float)):
        actual_cost = round(float(raw_cost), 4)
    else:
        raise RuntimeError("Real mode usage field 'cost_usd' must be numeric or null")

    provider_raw = raw.get("provider_raw")
    if not isinstance(provider_raw, dict):
        raise RuntimeError("Real mode usage field 'provider_raw' must be an object")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("Real mode usage field 'run_id' must be a non-empty string")

    usage = {
        "prefill_tokens": tokens.prefill,
        "sample_tokens": tokens.sample,
        "train_tokens": tokens.train,
        "cost_usd": raw_cost if raw_cost is None else round(float(raw_cost), 4),
        "provider_raw": provider_raw,
        "run_id": run_id,
    }
    return tokens, actual_cost, usage


def _mock_usage(stage: str, actual_tokens: TokenUsage, actual_cost: float) -> dict[str, object]:
    return {
        "prefill_tokens": actual_tokens.prefill,
        "sample_tokens": actual_tokens.sample,
        "train_tokens": actual_tokens.train,
        "cost_usd": round(actual_cost, 4),
        "provider_raw": {"mode": "mock", "stage": stage},
        "run_id": f"mock-{stage}",
    }


def _record_stage(
    *,
    stage: str,
    mode: str,
    paths: PipelinePaths,
    ledger: Ledger,
    projected_tokens: TokenUsage,
    projected_cost: float,
    actual_tokens: TokenUsage,
    actual_cost: float,
) -> tuple[StageRecord, Ledger]:
    record = StageRecord(
        mode=mode,
        stage=stage,
        projected_cost_usd=projected_cost,
        actual_cost_usd=actual_cost,
        projected_tokens=projected_tokens,
        actual_tokens=actual_tokens,
        status="completed",
    )
    updated = add_record(ledger, record)
    save_ledger(paths.ledger, updated)
    return record, updated


def _payload_list(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def _write_stage_audit(
    *,
    cfg: ProjectConfig,
    paths: PipelinePaths,
    preflight: PreflightResult,
    mode: str,
    stage: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    ledger_before: Ledger,
    ledger_after: Ledger,
    projected_tokens: TokenUsage,
    projected_cost: float,
    actual_tokens: TokenUsage,
    actual_cost: float,
    usage: dict[str, object],
    payload_for_audit: dict[str, object],
    prompt_traces: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
) -> None:
    stage_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "stage": stage,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 4),
        "projected_cost_usd": round(projected_cost, 4),
        "actual_cost_usd": round(actual_cost, 4),
        "stage_cap_usd": cfg.budget.stage_budgets_usd[stage],
        "cumulative_total_before_usd": round(ledger_before.total_spend_usd, 4),
        "cumulative_total_after_usd": round(ledger_after.total_spend_usd, 4),
        "projected_tokens": projected_tokens.as_dict(),
        "actual_tokens": actual_tokens.as_dict(),
        "usage": usage,
        "payload": payload_for_audit,
        "prompt_traces": prompt_traces,
    }
    if eval_rows:
        stage_payload["eval_rows_count"] = len(eval_rows)

    audit.write_stage_audit(paths.stage_audit(stage), stage_payload)
    if stage == "eval":
        audit.write_eval_rows(paths.eval_rows, eval_rows)

    audit.upsert_run_manifest(
        path=paths.run_manifest,
        cfg=cfg,
        mode=mode,
        state_dir=paths.root,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        warnings=preflight.warnings,
    )


def run_rl(
    cfg: ProjectConfig,
    paths: PipelinePaths,
    *,
    adapter: RLStageAdapter,
    mode: str,
    preflight: PreflightResult,
) -> None:
    stage = "rl"
    started_at = audit.utc_now_iso()
    started_monotonic = time.monotonic()
    ledger, projected_tokens, projected_cost = _stage_budget_check(stage, cfg, paths)
    _, _, mock_actual_tokens, mock_actual_cost = _projected_and_actual_mock(stage, cfg)
    payload = adapter.run(cfg=cfg, actual_cost_usd=mock_actual_cost)
    payload_for_audit = copy.deepcopy(payload)
    if mode == "real":
        actual_tokens, actual_cost, usage = _real_usage_from_payload(payload, cfg)
    else:
        actual_tokens, actual_cost = mock_actual_tokens, mock_actual_cost
        usage = _mock_usage(stage, actual_tokens, actual_cost)
    _, updated_ledger = _record_stage(
        stage=stage,
        mode=mode,
        paths=paths,
        ledger=ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
    finished_at = audit.utc_now_iso()
    prompt_traces = _payload_list(payload_for_audit, PROMPT_TRACES_KEY)
    eval_rows = _payload_list(payload_for_audit, EVAL_ROWS_KEY)
    _write_stage_audit(
        cfg=cfg,
        paths=paths,
        preflight=preflight,
        mode=mode,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started_monotonic,
        ledger_before=ledger,
        ledger_after=updated_ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        usage=usage,
        payload_for_audit=payload_for_audit,
        prompt_traces=prompt_traces,
        eval_rows=eval_rows,
    )
    payload.pop(REAL_USAGE_KEY, None)
    payload.pop(PROMPT_TRACES_KEY, None)
    payload.pop(EVAL_ROWS_KEY, None)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_teacher_checkpoint(payload)
    _write_json(paths.teacher_ckpt, payload)


def run_fsdp(
    cfg: ProjectConfig,
    paths: PipelinePaths,
    *,
    adapter: FSDPStageAdapter,
    mode: str,
    preflight: PreflightResult,
) -> None:
    stage = "fsdp"
    started_at = audit.utc_now_iso()
    started_monotonic = time.monotonic()
    ledger, projected_tokens, projected_cost = _stage_budget_check(stage, cfg, paths)
    _, _, mock_actual_tokens, mock_actual_cost = _projected_and_actual_mock(stage, cfg)
    teacher = _read_json(paths.teacher_ckpt)
    payload = adapter.run(cfg=cfg, teacher_payload=teacher, actual_cost_usd=mock_actual_cost)
    payload_for_audit = copy.deepcopy(payload)
    if mode == "real":
        actual_tokens, actual_cost, usage = _real_usage_from_payload(payload, cfg)
    else:
        actual_tokens, actual_cost = mock_actual_tokens, mock_actual_cost
        usage = _mock_usage(stage, actual_tokens, actual_cost)
    _, updated_ledger = _record_stage(
        stage=stage,
        mode=mode,
        paths=paths,
        ledger=ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
    finished_at = audit.utc_now_iso()
    prompt_traces = _payload_list(payload_for_audit, PROMPT_TRACES_KEY)
    eval_rows = _payload_list(payload_for_audit, EVAL_ROWS_KEY)
    _write_stage_audit(
        cfg=cfg,
        paths=paths,
        preflight=preflight,
        mode=mode,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started_monotonic,
        ledger_before=ledger,
        ledger_after=updated_ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        usage=usage,
        payload_for_audit=payload_for_audit,
        prompt_traces=prompt_traces,
        eval_rows=eval_rows,
    )
    payload.pop(REAL_USAGE_KEY, None)
    payload.pop(PROMPT_TRACES_KEY, None)
    payload.pop(EVAL_ROWS_KEY, None)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_teacher_checkpoint(payload)
    _write_json(paths.teacher_ckpt, payload)


def run_distill(
    cfg: ProjectConfig,
    paths: PipelinePaths,
    *,
    adapter: DistillStageAdapter,
    mode: str,
    preflight: PreflightResult,
) -> None:
    stage = "distill"
    started_at = audit.utc_now_iso()
    started_monotonic = time.monotonic()
    ledger, projected_tokens, projected_cost = _stage_budget_check(stage, cfg, paths)
    _, _, mock_actual_tokens, mock_actual_cost = _projected_and_actual_mock(stage, cfg)
    teacher = _read_json(paths.teacher_ckpt)
    payload = adapter.run(cfg=cfg, teacher_payload=teacher, actual_cost_usd=mock_actual_cost)
    payload_for_audit = copy.deepcopy(payload)
    if mode == "real":
        actual_tokens, actual_cost, usage = _real_usage_from_payload(payload, cfg)
    else:
        actual_tokens, actual_cost = mock_actual_tokens, mock_actual_cost
        usage = _mock_usage(stage, actual_tokens, actual_cost)
    _, updated_ledger = _record_stage(
        stage=stage,
        mode=mode,
        paths=paths,
        ledger=ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
    finished_at = audit.utc_now_iso()
    prompt_traces = _payload_list(payload_for_audit, PROMPT_TRACES_KEY)
    eval_rows = _payload_list(payload_for_audit, EVAL_ROWS_KEY)
    _write_stage_audit(
        cfg=cfg,
        paths=paths,
        preflight=preflight,
        mode=mode,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started_monotonic,
        ledger_before=ledger,
        ledger_after=updated_ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        usage=usage,
        payload_for_audit=payload_for_audit,
        prompt_traces=prompt_traces,
        eval_rows=eval_rows,
    )
    payload.pop(REAL_USAGE_KEY, None)
    payload.pop(PROMPT_TRACES_KEY, None)
    payload.pop(EVAL_ROWS_KEY, None)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_student_checkpoint(payload)
    _write_json(paths.student_ckpt, payload)


def run_eval(
    cfg: ProjectConfig,
    paths: PipelinePaths,
    *,
    adapter: EvalStageAdapter,
    mode: str,
    preflight: PreflightResult,
) -> None:
    stage = "eval"
    started_at = audit.utc_now_iso()
    started_monotonic = time.monotonic()
    ledger, projected_tokens, projected_cost = _stage_budget_check(stage, cfg, paths)
    _, _, mock_actual_tokens, mock_actual_cost = _projected_and_actual_mock(stage, cfg)
    teacher = _read_json(paths.teacher_ckpt)
    student = _read_json(paths.student_ckpt)
    payload = adapter.run(
        cfg=cfg,
        teacher_payload=teacher,
        student_payload=student,
        actual_cost_usd=mock_actual_cost,
    )
    payload_for_audit = copy.deepcopy(payload)
    if mode == "real":
        actual_tokens, actual_cost, usage = _real_usage_from_payload(payload, cfg)
    else:
        actual_tokens, actual_cost = mock_actual_tokens, mock_actual_cost
        usage = _mock_usage(stage, actual_tokens, actual_cost)
    _, updated_ledger = _record_stage(
        stage=stage,
        mode=mode,
        paths=paths,
        ledger=ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
    finished_at = audit.utc_now_iso()
    prompt_traces = _payload_list(payload_for_audit, PROMPT_TRACES_KEY)
    eval_rows = _payload_list(payload_for_audit, EVAL_ROWS_KEY)
    _write_stage_audit(
        cfg=cfg,
        paths=paths,
        preflight=preflight,
        mode=mode,
        stage=stage,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started_monotonic,
        ledger_before=ledger,
        ledger_after=updated_ledger,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
        usage=usage,
        payload_for_audit=payload_for_audit,
        prompt_traces=prompt_traces,
        eval_rows=eval_rows,
    )
    payload.pop(REAL_USAGE_KEY, None)
    payload.pop(PROMPT_TRACES_KEY, None)
    payload.pop(EVAL_ROWS_KEY, None)
    if isinstance(payload.get("cost"), dict):
        payload["cost"]["eval_stage_cost_usd"] = round(actual_cost, 4)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_eval_metrics(payload)
    _write_json(paths.eval_metrics, payload)


def run_report(cfg: ProjectConfig, paths: PipelinePaths, *, mode: str, preflight: PreflightResult) -> None:
    ledger = load_ledger(paths.ledger)
    eval_metrics = _read_json(paths.eval_metrics)
    text = _format_report(cfg, ledger, eval_metrics, mode=mode, preflight=preflight)
    paths.report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text(text)
    now = audit.utc_now_iso()
    audit.upsert_run_manifest(
        path=paths.run_manifest,
        cfg=cfg,
        mode=mode,
        state_dir=paths.root,
        stage="report",
        started_at=now,
        finished_at=now,
        warnings=preflight.warnings,
    )
    manifest = audit.load_json(paths.run_manifest)
    stage_payloads = [audit.load_json(paths.stage_audit(stage)) for stage in REQUIRED_STAGES if paths.stage_audit(stage).exists()]
    eval_rows = audit.load_eval_rows(paths.eval_rows)
    audit_report_text = audit.format_run_audit_report(
        cfg=cfg,
        mode=mode,
        manifest=manifest,
        stage_payloads=stage_payloads,
        eval_rows=eval_rows,
    )
    paths.audit_report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.audit_report_md.write_text(audit_report_text)


def _format_report(
    cfg: ProjectConfig,
    ledger: Ledger,
    eval_metrics: dict[str, object],
    *,
    mode: str,
    preflight: PreflightResult,
) -> str:
    quality = eval_metrics["quality"]
    benchmark = quality["benchmark"]
    llm_judge = quality["llm_judge"]
    cost = eval_metrics["cost"]
    infer = cost["inference_usd_per_1k_tokens"]
    stability = eval_metrics["training_stability"]
    projected_total = round(sum(record.projected_cost_usd for record in ledger.records), 4)
    projected_prefill = sum(record.projected_tokens.prefill for record in ledger.records)
    projected_sample = sum(record.projected_tokens.sample for record in ledger.records)
    projected_train = sum(record.projected_tokens.train for record in ledger.records)
    setup_status = "ready" if preflight.ok else "blocked"

    lines = [
        f"# {cfg.name} Eval Report",
        "",
        "## Run Metadata",
        f"- Run mode: {mode}",
        f"- Setup status: {setup_status}",
        f"- Schema version: {SCHEMA_VERSION}",
        "",
        "## Disclaimer",
    ]

    if mode == "mock":
        lines.append("- This report was generated in mock mode with deterministic stage adapters for showcase workflow validation.")
    else:
        lines.append("- This report was generated in real mode using external services configured by your environment.")

    for warning in preflight.warnings:
        lines.append(f"- Preflight warning: {warning}")

    lines.extend(
        [
            "",
            "## Quality",
            f"- Baseline benchmark score: {benchmark['baseline']}",
            f"- Teacher benchmark score: {benchmark['teacher']}",
            f"- Student benchmark score: {benchmark['student']}",
            f"- Student retention vs teacher: {benchmark['student_retention_vs_teacher']}",
            f"- LLM judge win rate (student vs baseline): {llm_judge['student_vs_baseline_win_rate']}",
            f"- LLM judge win rate (student vs teacher): {llm_judge['student_vs_teacher_win_rate']}",
            "",
            "## Cost",
            f"- Tinker target cap (USD): {cfg.budget.target_cap_usd:.2f}",
            f"- Tinker hard cap (USD): {cfg.budget.hard_cap_usd:.2f}",
            f"- Projected spend (USD): {projected_total:.2f}",
            f"- Actual spend (USD): {ledger.total_spend_usd:.2f}",
            f"- Teacher inference cost / 1k tokens (USD): {infer['teacher']}",
            f"- Student inference cost / 1k tokens (USD): {infer['student']}",
            f"- Student inference savings (%): {infer['student_savings_pct']}",
            f"- Projected token totals: prefill={projected_prefill}, sample={projected_sample}, train={projected_train}",
            f"- Actual token totals: prefill={ledger.token_totals.prefill}, sample={ledger.token_totals.sample}, train={ledger.token_totals.train}",
            "",
            "## Training Stability",
            f"- RL stability score: {stability['rl']['stability_score']} (NaN events: {stability['rl']['nan_events']})",
            f"- FSDP stability score: {stability['fsdp']['stability_score']} (NaN events: {stability['fsdp']['nan_events']})",
            f"- Distill stability score: {stability['distill']['stability_score']} (NaN events: {stability['distill']['nan_events']})",
            "",
            "## Stage Spend",
        ]
    )
    for stage in ("rl", "fsdp", "distill", "eval"):
        lines.append(f"- {stage}: ${ledger.stage_spend_usd.get(stage, 0.0):.2f}")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return json.loads(path.read_text())
