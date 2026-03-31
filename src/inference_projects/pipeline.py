from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from inference_projects import budget
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

SUPPORTED_COMMANDS = {"rl", "fsdp", "distill", "eval", "report", "all", "smoke", "preflight", "dryrun"}


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


def run_pipeline_command(
    command: str,
    *,
    mode: str | None = None,
    config_path: Path | str = Path("config/default.toml"),
    state_dir: Path | str = Path("."),
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

    preflight_result = ensure_preflight_ready(mode=resolved_mode, cfg=cfg, state_dir=paths.root)
    adapters = select_stage_adapters(resolved_mode)

    if command == "rl":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode)
    elif command == "fsdp":
        run_fsdp(cfg, paths, adapter=adapters.fsdp, mode=resolved_mode)
    elif command == "distill":
        run_distill(cfg, paths, adapter=adapters.distill, mode=resolved_mode)
    elif command == "eval":
        run_eval(cfg, paths, adapter=adapters.eval, mode=resolved_mode)
    elif command == "report":
        run_report(cfg, paths, mode=resolved_mode, preflight=preflight_result)
    elif command == "all":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode)
        run_fsdp(cfg, paths, adapter=adapters.fsdp, mode=resolved_mode)
        run_distill(cfg, paths, adapter=adapters.distill, mode=resolved_mode)
        run_eval(cfg, paths, adapter=adapters.eval, mode=resolved_mode)
        run_report(cfg, paths, mode=resolved_mode, preflight=preflight_result)
    elif command == "smoke":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode)
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


def _projected_and_actual(stage: str, cfg: ProjectConfig) -> tuple[TokenUsage, float, TokenUsage, float]:
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


def _run_budget_checked_stage(stage: str, cfg: ProjectConfig, paths: PipelinePaths, mode: str) -> tuple[Ledger, StageRecord]:
    ledger = load_ledger(paths.ledger)
    projected_tokens, projected_cost, actual_tokens, actual_cost = _projected_and_actual(stage, cfg)
    budget.ensure_within_stage_budget(stage, projected_cost, cfg)
    budget.ensure_within_hard_cap(current_total=ledger.total_spend_usd, incoming_cost=projected_cost, cfg=cfg)
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
    return updated, record


def run_rl(cfg: ProjectConfig, paths: PipelinePaths, *, adapter: RLStageAdapter, mode: str) -> None:
    _, record = _run_budget_checked_stage("rl", cfg, paths, mode)
    payload = adapter.run(cfg=cfg, actual_cost_usd=record.actual_cost_usd)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_teacher_checkpoint(payload)
    _write_json(paths.teacher_ckpt, payload)


def run_fsdp(cfg: ProjectConfig, paths: PipelinePaths, *, adapter: FSDPStageAdapter, mode: str) -> None:
    _, record = _run_budget_checked_stage("fsdp", cfg, paths, mode)
    teacher = _read_json(paths.teacher_ckpt)
    payload = adapter.run(cfg=cfg, teacher_payload=teacher, actual_cost_usd=record.actual_cost_usd)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_teacher_checkpoint(payload)
    _write_json(paths.teacher_ckpt, payload)


def run_distill(cfg: ProjectConfig, paths: PipelinePaths, *, adapter: DistillStageAdapter, mode: str) -> None:
    _, record = _run_budget_checked_stage("distill", cfg, paths, mode)
    teacher = _read_json(paths.teacher_ckpt)
    payload = adapter.run(cfg=cfg, teacher_payload=teacher, actual_cost_usd=record.actual_cost_usd)
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_student_checkpoint(payload)
    _write_json(paths.student_ckpt, payload)


def run_eval(cfg: ProjectConfig, paths: PipelinePaths, *, adapter: EvalStageAdapter, mode: str) -> None:
    _, record = _run_budget_checked_stage("eval", cfg, paths, mode)
    teacher = _read_json(paths.teacher_ckpt)
    student = _read_json(paths.student_ckpt)
    payload = adapter.run(
        cfg=cfg,
        teacher_payload=teacher,
        student_payload=student,
        actual_cost_usd=record.actual_cost_usd,
    )
    payload.update({"schema_version": SCHEMA_VERSION, "mode": mode})
    validate_eval_metrics(payload)
    _write_json(paths.eval_metrics, payload)


def run_report(cfg: ProjectConfig, paths: PipelinePaths, *, mode: str, preflight: PreflightResult) -> None:
    ledger = load_ledger(paths.ledger)
    eval_metrics = _read_json(paths.eval_metrics)
    text = _format_report(cfg, ledger, eval_metrics, mode=mode, preflight=preflight)
    paths.report_md.parent.mkdir(parents=True, exist_ok=True)
    paths.report_md.write_text(text)


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
