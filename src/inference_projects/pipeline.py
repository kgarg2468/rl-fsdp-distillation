from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from inference_projects import audit
from inference_projects import budget
from inference_projects import campaign as campaign_utils
from inference_projects import tuning as tuning_utils
from inference_projects.adapters import (
    DistillStageAdapter,
    EvalStageAdapter,
    StageAdapters,
    TeacherFTStageAdapter,
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

SUPPORTED_COMMANDS = {
    "rl",
    "teacher_ft",
    "distill",
    "eval",
    "report",
    "all",
    "smoke",
    "preflight",
    "dryrun",
    "campaign",
    "tune",
}


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


@dataclass(frozen=True)
class ExecutionOptions:
    resume: bool
    heartbeat_seconds: int
    progress_timeout_seconds: float


@dataclass
class StageExecutionContext:
    command: str
    run_id: str
    stage: str
    last_checkpoint_ts: str = field(default_factory=audit.utc_now_iso)


class StageExecutionError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str):
        super().__init__(message)
        self.failure_class = failure_class


class GuardrailViolationError(RuntimeError):
    def __init__(self, message: str, *, failure_class: str):
        super().__init__(message)
        self.failure_class = failure_class


def run_pipeline_command(
    command: str,
    *,
    mode: str | None = None,
    config_path: Path | str = Path("config/default.toml"),
    state_dir: Path | str = Path("."),
    resume: bool = True,
    heartbeat_seconds: int = 30,
    progress_timeout_seconds: float | None = None,
) -> dict[str, object] | None:
    if command not in SUPPORTED_COMMANDS:
        raise ValueError(f"Unknown command: {command}")

    cfg = load_config(config_path)
    resolved_mode = mode or cfg.runtime.default_mode
    paths = PipelinePaths(Path(state_dir))
    options = ExecutionOptions(
        resume=resume,
        heartbeat_seconds=max(1, int(heartbeat_seconds)),
        progress_timeout_seconds=(
            float(progress_timeout_seconds)
            if progress_timeout_seconds is not None
            else float(cfg.runtime.retry_progress_timeout_seconds)
        ),
    )

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
            options=options,
        )

    if command == "tune":
        return run_tune(
            cfg=cfg,
            mode=resolved_mode,
            state_dir=paths.root,
            options=options,
        )

    preflight_result = ensure_preflight_ready(mode=resolved_mode, cfg=cfg, state_dir=paths.root)
    adapters = select_stage_adapters(resolved_mode)

    if command == "rl":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode, preflight=preflight_result)
    elif command == "teacher_ft":
        run_teacher_ft(cfg, paths, adapter=adapters.teacher_ft, mode=resolved_mode, preflight=preflight_result)
    elif command == "distill":
        run_distill(cfg, paths, adapter=adapters.distill, mode=resolved_mode, preflight=preflight_result)
    elif command == "eval":
        run_eval(cfg, paths, adapter=adapters.eval, mode=resolved_mode, preflight=preflight_result)
    elif command == "report":
        run_report(cfg, paths, mode=resolved_mode, preflight=preflight_result)
    elif command == "all":
        _run_all_with_reliability(
            cfg=cfg,
            mode=resolved_mode,
            paths=paths,
            options=options,
            preflight_result=preflight_result,
            adapters=adapters,
        )
    elif command == "smoke":
        run_rl(cfg, paths, adapter=adapters.rl, mode=resolved_mode, preflight=preflight_result)
    return None


def _dryrun_summary(cfg: ProjectConfig, mode: str) -> dict[str, object]:
    preflight = run_preflight(mode=mode, cfg=cfg, check_state_dir=False)
    stages: dict[str, dict[str, object]] = {}
    running = 0.0
    for stage in REQUIRED_STAGES:
        projected_cost = budget.projected_stage_cost_usd(stage, cfg)
        running += projected_cost
        stages[stage] = {
            "projected_cost_usd": projected_cost,
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


def _emit_log(event: str, **fields: object) -> None:
    payload = {"ts": audit.utc_now_iso(), "event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _read_run_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": {}, "status": "new"}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return {"completed": {}, "status": "invalid"}
    completed = raw.get("completed")
    if not isinstance(completed, dict):
        raw["completed"] = {}
    return raw


def _write_run_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _mark_completed_stage(state: dict[str, Any], *, run_id: str, stage: str) -> None:
    completed = state.setdefault("completed", {})
    if not isinstance(completed, dict):
        completed = {}
        state["completed"] = completed
    run_stages = completed.setdefault(run_id, [])
    if not isinstance(run_stages, list):
        run_stages = []
        completed[run_id] = run_stages
    if stage not in run_stages:
        run_stages.append(stage)


def _is_stage_completed(state: dict[str, Any], *, run_id: str, stage: str) -> bool:
    completed = state.get("completed", {})
    if not isinstance(completed, dict):
        return False
    run_stages = completed.get(run_id, [])
    return isinstance(run_stages, list) and stage in run_stages


def _classify_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return "stalled"
    if isinstance(exc, FileNotFoundError):
        return "invariant_failed"
    if isinstance(exc, GuardrailViolationError):
        return exc.failure_class
    if isinstance(exc, StageExecutionError):
        return exc.failure_class
    if "budget cap" in message or "token cap" in message:
        return "budget_cap_exceeded"
    if "run cap" in message:
        return "run_cap_reached"
    if "reproducibility" in message:
        return "reproducibility_failed"
    if "integrity" in message:
        return "integrity_failed"
    if "retry" in message or "timeout" in message or "connection" in message:
        return "transient_exhausted"
    return "failed"


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(cfg: ProjectConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": cfg.name,
        "seed": cfg.seed,
        "models": {
            "teacher": cfg.teacher_model,
            "student": cfg.student_model,
            "baseline": cfg.baseline_model,
        },
        "token_rates_per_million": dict(sorted(cfg.token_rates_per_million.items())),
        "token_caps": {stage: cfg.token_caps[stage].as_dict() for stage in REQUIRED_STAGES},
        "runtime": {
            "default_mode": cfg.runtime.default_mode,
            "projection_warning_min_usd": cfg.runtime.projection_warning_min_usd,
            "projection_warning_max_usd": cfg.runtime.projection_warning_max_usd,
            "real_required_env": list(cfg.runtime.real_required_env),
            "real_poll_interval_seconds": cfg.runtime.real_poll_interval_seconds,
            "real_poll_timeout_seconds": cfg.runtime.real_poll_timeout_seconds,
            "retry_max_connections": cfg.runtime.retry_max_connections,
            "retry_progress_timeout_seconds": cfg.runtime.retry_progress_timeout_seconds,
            "retry_delay_base_seconds": cfg.runtime.retry_delay_base_seconds,
            "retry_delay_max_seconds": cfg.runtime.retry_delay_max_seconds,
            "retry_jitter_factor": cfg.runtime.retry_jitter_factor,
            "retry_enabled": cfg.runtime.retry_enabled,
            "max_consecutive_failures": cfg.runtime.max_consecutive_failures,
        },
        "evaluation": {
            "prompt_file": str(cfg.evaluation.prompt_file),
            "prompt_limit": cfg.evaluation.prompt_limit,
            "max_concurrency": cfg.evaluation.max_concurrency,
            "batch_size": cfg.evaluation.batch_size,
            "max_tokens_eval": cfg.evaluation.max_tokens_eval,
            "eval_temperature": cfg.evaluation.eval_temperature,
            "eval_stop_tokens": list(cfg.evaluation.eval_stop_tokens),
            "eval_max_tokens_candidates": list(cfg.evaluation.eval_max_tokens_candidates),
            "teacher_integrity_refusal_threshold": cfg.evaluation.teacher_integrity_refusal_threshold,
            "teacher_integrity_min_score": cfg.evaluation.teacher_integrity_min_score,
            "teacher_integrity_numeric_parse_threshold": cfg.evaluation.teacher_integrity_numeric_parse_threshold,
        },
        "distillation": {
            "training_prompt_limit": cfg.distillation.training_prompt_limit,
            "training_prompt_file": (
                str(cfg.distillation.training_prompt_file) if cfg.distillation.training_prompt_file is not None else None
            ),
            "teacher_prompt_template": cfg.distillation.teacher_prompt_template,
            "filter_profile": cfg.distillation.filter_profile,
            "hard_example_ratio": cfg.distillation.hard_example_ratio,
            "kd_alpha": cfg.distillation.kd_alpha,
            "kd_temperature": cfg.distillation.kd_temperature,
            "learning_rate": cfg.distillation.learning_rate,
            "epochs": cfg.distillation.epochs,
            "batch_size": cfg.distillation.batch_size,
            "warmup_ratio": cfg.distillation.warmup_ratio,
            "lora_rank": cfg.distillation.lora_rank,
            "context_length": cfg.distillation.context_length,
            "grad_clip": cfg.distillation.grad_clip,
            "weight_decay": cfg.distillation.weight_decay,
        },
        "campaign": {
            "seeds": list(cfg.campaign.seeds),
            "min_runs": cfg.campaign.min_runs,
            "max_runs": cfg.campaign.max_runs,
            "bootstrap_reps": cfg.campaign.bootstrap_reps,
            "early_stop_threshold": cfg.campaign.early_stop_threshold,
            "strict_run_cap": cfg.campaign.strict_run_cap,
        },
        "tuning": {
            "stage1_prompt_limit": cfg.tuning.stage1_prompt_limit,
            "stage2_prompt_limit": cfg.tuning.stage2_prompt_limit,
            "sweep_runs": cfg.tuning.sweep_runs,
            "teacher_candidates": list(cfg.tuning.teacher_candidates),
            "promotion_top_k": cfg.tuning.promotion_top_k,
            "min_teacher_margin": cfg.tuning.min_teacher_margin,
            "min_student_gain": cfg.tuning.min_student_gain,
            "min_student_exact_gain": cfg.tuning.min_student_exact_gain,
            "min_student_numeric_parse": cfg.tuning.min_student_numeric_parse,
            "max_eval_duration_seconds": cfg.tuning.max_eval_duration_seconds,
            "strict_run_cap": cfg.tuning.strict_run_cap,
        },
    }
    return _sha256_text(_stable_json_dumps(payload))


def _minimal_campaign_run_fingerprint(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics", {})
    means = metrics.get("means", {}) if isinstance(metrics, dict) else {}
    return {
        "seed": run.get("seed"),
        "stages_completed": list(run.get("stages_completed", [])),
        "integrity": {
            "status": run.get("integrity", {}).get("status"),
            "passed": run.get("integrity", {}).get("passed"),
        },
        "acceptance_checks": dict(run.get("acceptance_checks", {})),
        "means": dict(means) if isinstance(means, dict) else {},
        "eval_rows_sha256": run.get("eval_rows_sha256", ""),
    }


def _minimal_tune_candidate_fingerprint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id", ""),
        "phase": row.get("phase", ""),
        "teacher_model": row.get("teacher_model", ""),
        "teacher_minus_baseline": row.get("teacher_minus_baseline", -999.0),
        "student_minus_baseline": row.get("student_minus_baseline", -999.0),
        "student_minus_baseline_exact_match": row.get("student_minus_baseline_exact_match", -999.0),
        "student_numeric_parse_rate": row.get("student_numeric_parse_rate", -999.0),
        "teacher_margin_pass": bool(row.get("teacher_margin_pass", False)),
        "student_gain_pass": bool(row.get("student_gain_pass", False)),
        "student_exact_gain_pass": bool(row.get("student_exact_gain_pass", False)),
        "student_numeric_parse_pass": bool(row.get("student_numeric_parse_pass", False)),
        "eval_runtime_pass": bool(row.get("eval_runtime_pass", False)),
        "integrity_pass": bool(row.get("integrity_pass", False)),
        "composite_pass": bool(row.get("composite_pass", False)),
    }


def _campaign_run_invariant(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": run.get("seed"),
        "stages_completed": list(run.get("stages_completed", [])),
        "has_metrics": bool("metrics" in run),
        "eval_row_ids_sha256": str(run.get("eval_row_ids_sha256", "")),
    }


def _enforce_reproducibility(
    *,
    command: str,
    mode: str,
    summary_path: Path,
    reproducibility_payload: dict[str, Any],
) -> None:
    if mode not in {"mock", "real"} or not summary_path.exists():
        return
    try:
        prior = json.loads(summary_path.read_text())
    except Exception:
        return
    if not isinstance(prior, dict):
        return
    if str(prior.get("campaign_status", "")) == "failed" or str(prior.get("status", "")) == "failed":
        return
    prior_repro = prior.get("reproducibility")
    if not isinstance(prior_repro, dict):
        return
    if str(prior_repro.get("mode", "")) != mode:
        return
    if str(prior_repro.get("input_fingerprint", "")) != str(reproducibility_payload.get("input_fingerprint", "")):
        return
    if mode == "mock":
        prior_artifact_fingerprint = str(prior_repro.get("artifact_fingerprint", ""))
        artifact_fingerprint = str(reproducibility_payload.get("artifact_fingerprint", ""))
        if (
            prior_artifact_fingerprint
            and artifact_fingerprint
            and prior_artifact_fingerprint != artifact_fingerprint
        ):
            raise GuardrailViolationError(
                f"Mock reproducibility mismatch for command '{command}': repeated execution with identical inputs produced different outputs.",
                failure_class="reproducibility_failed",
            )
        return

    prior_invariant_fingerprint = str(prior_repro.get("invariant_fingerprint", ""))
    invariant_fingerprint = str(reproducibility_payload.get("invariant_fingerprint", ""))
    # Backwards compatibility for older summaries that predate real-mode reproducibility enforcement.
    if not prior_invariant_fingerprint or not invariant_fingerprint:
        return
    if prior_invariant_fingerprint != invariant_fingerprint:
        raise GuardrailViolationError(
            f"Real-mode reproducibility mismatch for command '{command}': repeated execution with identical inputs violated procedure-level invariants.",
            failure_class="reproducibility_failed",
        )


def _run_with_watchdog(
    *,
    fn: Callable[[], None],
    context: StageExecutionContext,
    options: ExecutionOptions,
) -> None:
    result: dict[str, object] = {}

    def _runner() -> None:
        try:
            fn()
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - surfaced by caller
            result["exception"] = exc

    started = time.monotonic()
    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    last_heartbeat = started
    poll_interval = min(0.2, max(0.01, options.progress_timeout_seconds / 10.0))
    while thread.is_alive():
        now = time.monotonic()
        elapsed = now - started
        if elapsed > options.progress_timeout_seconds:
            raise TimeoutError(
                f"Stage stalled: command={context.command} run={context.run_id} stage={context.stage} "
                f"elapsed={round(elapsed, 2)}s timeout={options.progress_timeout_seconds}s"
            )
        if now - last_heartbeat >= options.heartbeat_seconds:
            _emit_log(
                "heartbeat",
                command=context.command,
                run=context.run_id,
                stage=context.stage,
                elapsed_seconds=round(elapsed, 2),
                last_checkpoint_ts=context.last_checkpoint_ts,
            )
            last_heartbeat = now
        thread.join(timeout=poll_interval)
    if "exception" in result:
        raise result["exception"]  # type: ignore[misc]


def _normalize_integrity_status(raw: object) -> str:
    status = str(raw or "pass")
    if status == "ok":
        return "pass"
    if status == "integrity_failed":
        return "fail"
    return status


def _read_eval_integrity(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    integrity = payload.get("integrity", {})
    if not isinstance(integrity, dict):
        integrity = {}
    status = _normalize_integrity_status(integrity.get("status", "pass"))
    passed = bool(integrity.get("passed", status == "pass")) and status == "pass"
    return {
        "status": status,
        "passed": passed,
        "reason": str(integrity.get("reason", "")),
        "checks": dict(integrity.get("checks", {})) if isinstance(integrity.get("checks"), dict) else {},
    }


def _persist_stage_failure(
    *,
    paths: PipelinePaths,
    command: str,
    run_id: str,
    stage: str,
    attempt: int,
    elapsed_seconds: float,
    exc: Exception,
    failure_class: str,
) -> None:
    payload = {
        "ts": audit.utc_now_iso(),
        "command": command,
        "run_id": run_id,
        "stage": stage,
        "attempt": attempt,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "failure_class": failure_class,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    paths.audit_dir.mkdir(parents=True, exist_ok=True)
    failures_path = paths.audit_dir / f"{command}_failures.jsonl"
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    latest_path = paths.audit_dir / f"{command}_failure_latest.json"
    latest_path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_all_with_reliability(
    *,
    cfg: ProjectConfig,
    mode: str,
    paths: PipelinePaths,
    options: ExecutionOptions,
    preflight_result: PreflightResult,
    adapters: StageAdapters,
) -> None:
    command = "all"
    run_id = "all"
    stages = [*REQUIRED_STAGES, "report"]
    run_state_path = paths.audit_dir / "all_run_state.json"
    run_state = _read_run_state(run_state_path)
    stage_durations: dict[str, float] = {}
    completed_stages: list[str] = []
    last_completed_stage = ""
    attempts = max(1, int(cfg.runtime.max_consecutive_failures))
    non_retryable = {
        "integrity_failed",
        "budget_cap_exceeded",
        "run_cap_reached",
        "reproducibility_failed",
        "invariant_failed",
    }

    _emit_log(
        "command_start",
        command=command,
        mode=mode,
        state_dir=str(paths.root),
        resume=options.resume,
        heartbeat_seconds=options.heartbeat_seconds,
        progress_timeout_seconds=options.progress_timeout_seconds,
    )

    def _stage_call(stage: str) -> None:
        if stage == "rl":
            run_rl(cfg, paths, adapter=adapters.rl, mode=mode, preflight=preflight_result)
        elif stage == "teacher_ft":
            run_teacher_ft(cfg, paths, adapter=adapters.teacher_ft, mode=mode, preflight=preflight_result)
        elif stage == "distill":
            run_distill(cfg, paths, adapter=adapters.distill, mode=mode, preflight=preflight_result)
        elif stage == "eval":
            run_eval(cfg, paths, adapter=adapters.eval, mode=mode, preflight=preflight_result)
        elif stage == "report":
            run_report(cfg, paths, mode=mode, preflight=preflight_result)

    try:
        for stage in stages:
            if options.resume and _is_stage_completed(run_state, run_id=run_id, stage=stage):
                _verify_resumed_stage(stage=stage, paths=paths)
                completed_stages.append(stage)
                _emit_log("stage_resume_skip", command=command, run=run_id, stage=stage)
                continue

            for attempt in range(1, attempts + 1):
                stage_started = time.monotonic()
                ledger_before = load_ledger(paths.ledger)
                records_before = len(ledger_before.records)
                _emit_log(
                    "stage_start",
                    command=command,
                    run=run_id,
                    stage=stage,
                    attempt=attempt,
                    max_attempts=attempts,
                )
                try:
                    _run_with_watchdog(
                        fn=lambda stage=stage: _stage_call(stage),
                        context=StageExecutionContext(command=command, run_id=run_id, stage=stage),
                        options=options,
                    )
                    stage_durations[stage] = round(time.monotonic() - stage_started, 4)
                    ledger_after = _verify_campaign_stage(
                        stage=stage,
                        paths=paths,
                        ledger_records_before=records_before,
                    )

                    if stage == "eval" and mode == "real":
                        integrity = _read_eval_integrity(paths.eval_metrics)
                        if not bool(integrity["passed"]):
                            raise GuardrailViolationError(
                                "Real-mode all run failed integrity gate: "
                                f"{integrity['reason'] or 'integrity status is not pass'}",
                                failure_class="integrity_failed",
                            )

                    completed_stages.append(stage)
                    last_completed_stage = stage
                    _mark_completed_stage(run_state, run_id=run_id, stage=stage)
                    run_state["status"] = "running"
                    _write_run_state(run_state_path, run_state)

                    spend_delta = 0.0
                    if stage in REQUIRED_STAGES:
                        delta = round(ledger_after.total_spend_usd - ledger_before.total_spend_usd, 4)
                        if delta < 0:
                            raise RuntimeError(f"All stage '{stage}' produced negative spend delta: {delta}")
                        spend_delta = delta
                    _emit_log(
                        "stage_done",
                        command=command,
                        run=run_id,
                        stage=stage,
                        duration_seconds=stage_durations[stage],
                        spend_delta_usd=round(spend_delta, 4),
                    )
                    break
                except Exception as exc:
                    elapsed = time.monotonic() - stage_started
                    failure_class = _classify_exception(exc)
                    _persist_stage_failure(
                        paths=paths,
                        command=command,
                        run_id=run_id,
                        stage=stage,
                        attempt=attempt,
                        elapsed_seconds=elapsed,
                        exc=exc,
                        failure_class=failure_class,
                    )
                    if attempt >= attempts or failure_class in non_retryable:
                        raise StageExecutionError(
                            f"All stage failed after retries: run={run_id} stage={stage} error={exc}",
                            failure_class=failure_class,
                        ) from exc
                    _emit_log(
                        "stage_retry",
                        command=command,
                        run=run_id,
                        stage=stage,
                        attempt=attempt,
                        error=str(exc),
                        failure_class=failure_class,
                    )
                    sleep_seconds = min(
                        cfg.runtime.retry_delay_max_seconds,
                        cfg.runtime.retry_delay_base_seconds * (2 ** (attempt - 1)),
                    )
                    time.sleep(sleep_seconds)
    except Exception as exc:
        failure_class = _classify_exception(exc) if not isinstance(exc, StageExecutionError) else exc.failure_class
        run_state["status"] = "failed"
        run_state["failure_class"] = failure_class
        run_state["stop_reason"] = str(exc)
        run_state["last_completed_stage"] = last_completed_stage
        _write_run_state(run_state_path, run_state)
        _emit_log(
            "command_failure",
            command=command,
            failure_class=failure_class,
            stop_reason=str(exc),
            last_completed_stage=last_completed_stage,
        )
        raise RuntimeError(str(exc)) from exc

    run_state["status"] = "completed"
    run_state["last_completed_stage"] = last_completed_stage
    run_state["stage_durations_seconds"] = stage_durations
    _write_run_state(run_state_path, run_state)
    _emit_log(
        "command_summary",
        command=command,
        status="completed",
        stages_completed=completed_stages,
        run_state_path=str(run_state_path),
    )


def _campaign_required_artifacts(paths: PipelinePaths, stage: str) -> list[Path]:
    if stage == "rl":
        return [paths.teacher_ckpt]
    if stage == "teacher_ft":
        return [paths.teacher_ckpt]
    if stage == "distill":
        return [paths.student_ckpt]
    if stage == "eval":
        return [paths.eval_metrics, paths.eval_rows]
    if stage == "report":
        return [paths.report_md, paths.audit_report_md]
    raise ValueError(f"Unknown stage for artifact verification: {stage}")


def _verify_campaign_stage(
    *,
    stage: str,
    paths: PipelinePaths,
    ledger_records_before: int,
) -> Ledger:
    for path in _campaign_required_artifacts(paths, stage):
        if not path.exists():
            raise FileNotFoundError(f"Campaign stage '{stage}' missing required artifact: {path}")

    if stage in REQUIRED_STAGES:
        stage_audit_path = paths.stage_audit(stage)
        if not stage_audit_path.exists():
            raise FileNotFoundError(f"Campaign stage '{stage}' missing stage audit payload: {stage_audit_path}")
        stage_audit_payload = audit.load_json(stage_audit_path)
        if stage_audit_payload.get("status") != "completed":
            raise RuntimeError(f"Campaign stage '{stage}' audit status is not completed: {stage_audit_path}")

        ledger_after = load_ledger(paths.ledger)
        if len(ledger_after.records) != ledger_records_before + 1:
            raise RuntimeError(
                f"Campaign stage '{stage}' expected ledger record increment by 1 "
                f"(before={ledger_records_before}, after={len(ledger_after.records)})."
            )
        return ledger_after

    return load_ledger(paths.ledger)


def _verify_resumed_stage(*, stage: str, paths: PipelinePaths) -> Ledger:
    for path in _campaign_required_artifacts(paths, stage):
        if not path.exists():
            raise FileNotFoundError(f"Resumed stage '{stage}' missing required artifact: {path}")
    if stage in REQUIRED_STAGES:
        stage_audit_path = paths.stage_audit(stage)
        if not stage_audit_path.exists():
            raise FileNotFoundError(f"Resumed stage '{stage}' missing stage audit payload: {stage_audit_path}")
        stage_audit_payload = audit.load_json(stage_audit_path)
        if stage_audit_payload.get("status") != "completed":
            raise RuntimeError(f"Resumed stage '{stage}' audit status is not completed: {stage_audit_path}")
        ledger = load_ledger(paths.ledger)
        if not any(record.stage == stage for record in ledger.records):
            raise RuntimeError(f"Resumed stage '{stage}' has no corresponding ledger record: {paths.ledger}")
        return ledger
    return load_ledger(paths.ledger)


def run_campaign(
    *,
    cfg: ProjectConfig,
    mode: str,
    state_dir: Path,
    options: ExecutionOptions,
) -> dict[str, object]:
    campaign_dir = state_dir / "campaign"
    run_state_path = campaign_dir / "run_state.json"
    run_state = _read_run_state(run_state_path)
    frozen_prompts_path = campaign_dir / "frozen_prompts.jsonl"

    new_spend_usd = 0.0
    stop_reason = ""
    run_rows: list[list[dict[str, Any]]] = []
    run_summaries: list[dict[str, Any]] = []
    early_stop = {"triggered": False, "evaluated_after_runs": 0, "checks": {}}
    campaign_status = "ok"
    failure_class = ""
    last_completed_stage = ""
    recoverable = False
    frozen_info_payload: dict[str, Any] = {}
    config_fingerprint = _config_fingerprint(cfg)
    strict_run_cap = int(cfg.campaign.strict_run_cap)
    strict_run_cap_enforced = strict_run_cap > 0
    effective_max_runs = min(cfg.campaign.max_runs, strict_run_cap) if strict_run_cap_enforced else cfg.campaign.max_runs
    planned_seeds = list(cfg.campaign.seeds)[:effective_max_runs]
    strict_run_cap_hit = strict_run_cap_enforced and cfg.campaign.max_runs > effective_max_runs
    summary_path = campaign_dir / "campaign_summary.json"
    report_path = campaign_dir / "campaign_report.md"
    _emit_log(
        "command_start",
        command="campaign",
        mode=mode,
        config_name=cfg.name,
        state_dir=str(state_dir),
        resume=options.resume,
        heartbeat_seconds=options.heartbeat_seconds,
        progress_timeout_seconds=options.progress_timeout_seconds,
        strict_run_cap_configured=strict_run_cap,
        strict_run_cap_effective=effective_max_runs,
        strict_run_cap_enforced=strict_run_cap_enforced,
    )

    try:
        frozen_info = campaign_utils.freeze_prompt_file(
            source_path=cfg.evaluation.prompt_file,
            frozen_path=frozen_prompts_path,
            prompt_limit=cfg.evaluation.prompt_limit,
        )
        frozen_info_payload = frozen_info.as_dict()

        for run_index, seed in enumerate(planned_seeds, start=1):
            run_id = f"seed-{seed}"
            run_dir = campaign_dir / "runs" / run_id
            run_paths = PipelinePaths(run_dir)
            run_cfg = replace(
                cfg,
                seed=seed,
                evaluation=replace(cfg.evaluation, prompt_file=frozen_prompts_path),
            )
            preflight_result = ensure_preflight_ready(mode=mode, cfg=run_cfg, state_dir=run_dir)
            adapters = select_stage_adapters(mode)

            stage_durations: dict[str, float] = {}
            completed_stages: list[str] = []
            run_halted = False
            _emit_log(
                "run_start",
                command="campaign",
                run=run_id,
                run_index=run_index,
                planned_runs=len(planned_seeds),
                seed=seed,
            )

            for stage in (*REQUIRED_STAGES, "report"):
                if options.resume and _is_stage_completed(run_state, run_id=run_id, stage=stage):
                    _verify_resumed_stage(stage=stage, paths=run_paths)
                    completed_stages.append(stage)
                    _emit_log("stage_resume_skip", command="campaign", run=run_id, stage=stage)
                    continue

                attempts = max(1, int(cfg.runtime.max_consecutive_failures))
                for attempt in range(1, attempts + 1):
                    stage_started = time.monotonic()
                    ledger_before = load_ledger(run_paths.ledger)
                    records_before = len(ledger_before.records)
                    _emit_log(
                        "stage_start",
                        command="campaign",
                        run=run_id,
                        stage=stage,
                        attempt=attempt,
                        max_attempts=attempts,
                    )

                    def _stage_call() -> None:
                        if stage == "rl":
                            run_rl(run_cfg, run_paths, adapter=adapters.rl, mode=mode, preflight=preflight_result)
                        elif stage == "teacher_ft":
                            run_teacher_ft(run_cfg, run_paths, adapter=adapters.teacher_ft, mode=mode, preflight=preflight_result)
                        elif stage == "distill":
                            run_distill(
                                run_cfg,
                                run_paths,
                                adapter=adapters.distill,
                                mode=mode,
                                preflight=preflight_result,
                            )
                        elif stage == "eval":
                            run_eval(run_cfg, run_paths, adapter=adapters.eval, mode=mode, preflight=preflight_result)
                        elif stage == "report":
                            run_report(run_cfg, run_paths, mode=mode, preflight=preflight_result)

                    try:
                        _run_with_watchdog(
                            fn=_stage_call,
                            context=StageExecutionContext(command="campaign", run_id=run_id, stage=stage),
                            options=options,
                        )
                        stage_durations[stage] = round(time.monotonic() - stage_started, 4)
                        ledger_after = _verify_campaign_stage(
                            stage=stage,
                            paths=run_paths,
                            ledger_records_before=records_before,
                        )
                        completed_stages.append(stage)
                        last_completed_stage = stage
                        _mark_completed_stage(run_state, run_id=run_id, stage=stage)
                        _write_run_state(run_state_path, run_state)

                        spend_delta = 0.0
                        if stage in REQUIRED_STAGES:
                            delta = round(ledger_after.total_spend_usd - ledger_before.total_spend_usd, 4)
                            if delta < 0:
                                raise RuntimeError(f"Campaign stage '{stage}' produced negative spend delta: {delta}")
                            new_spend_usd = round(new_spend_usd + delta, 4)
                            spend_delta = delta
                        _emit_log(
                            "stage_done",
                            command="campaign",
                            run=run_id,
                            stage=stage,
                            duration_seconds=stage_durations[stage],
                            spend_delta_usd=round(spend_delta, 4),
                        )
                        break
                    except Exception as exc:
                        if attempt >= attempts:
                            raise StageExecutionError(
                                f"Campaign stage failed after retries: run={run_id} stage={stage} error={exc}",
                                failure_class=_classify_exception(exc),
                            ) from exc
                        _emit_log(
                            "stage_retry",
                            command="campaign",
                            run=run_id,
                            stage=stage,
                            attempt=attempt,
                            error=str(exc),
                        )
                        sleep_seconds = min(
                            cfg.runtime.retry_delay_max_seconds,
                            cfg.runtime.retry_delay_base_seconds * (2 ** (attempt - 1)),
                        )
                        time.sleep(sleep_seconds)

            run_summary: dict[str, Any] = {
                "seed": seed,
                "run_dir": str(run_dir),
                "stages_completed": completed_stages,
                "stage_durations_seconds": stage_durations,
                "actual_spend_usd": round(load_ledger(run_paths.ledger).total_spend_usd, 4),
                "artifacts": {
                    "eval_report": str(run_paths.report_md),
                    "run_audit_report": str(run_paths.audit_report_md),
                    "eval_rows": str(run_paths.eval_rows),
                    "ledger": str(run_paths.ledger),
                },
            }

            if "eval" in completed_stages:
                eval_rows = audit.load_eval_rows(run_paths.eval_rows)
                if not eval_rows:
                    raise RuntimeError(f"Campaign seed {seed} did not emit eval rows: {run_paths.eval_rows}")
                run_rows.append(eval_rows)
                run_summary["eval_rows_sha256"] = _sha256_file(run_paths.eval_rows)
                run_summary["eval_row_ids_sha256"] = _sha256_text(
                    _stable_json_dumps([str(row.get("row_id", "")) for row in eval_rows])
                )
                run_summary["metrics"] = campaign_utils.summarize_eval_rows(
                    eval_rows=eval_rows,
                    bootstrap_reps=run_cfg.campaign.bootstrap_reps,
                    rng_seed=seed,
                )
                eval_metrics = _read_json(run_paths.eval_metrics)
                integrity = eval_metrics.get("integrity", {})
                integrity_status_raw = str(integrity.get("status", "pass"))
                if integrity_status_raw == "ok":
                    integrity_status_raw = "pass"
                if integrity_status_raw == "integrity_failed":
                    integrity_status_raw = "fail"
                integrity_passed = bool(integrity.get("passed", integrity_status_raw == "pass")) and integrity_status_raw == "pass"
                run_summary["integrity"] = {
                    "passed": integrity_passed,
                    "status": integrity_status_raw,
                    "reason": str(integrity.get("reason", "")),
                    "checks": dict(integrity.get("checks", {})) if isinstance(integrity.get("checks"), dict) else {},
                }
                teacher_minus_baseline = float(
                    run_summary["metrics"]["means"]["teacher"] - run_summary["metrics"]["means"]["baseline"]
                )
                student_minus_baseline = float(
                    run_summary["metrics"]["means"]["student"] - run_summary["metrics"]["means"]["baseline"]
                )
                _emit_log(
                    "integrity_result",
                    command="campaign",
                    run=run_id,
                    integrity_status=integrity_status_raw,
                    teacher_minus_baseline=round(teacher_minus_baseline, 6),
                    student_minus_baseline=round(student_minus_baseline, 6),
                )
                acceptance_checks = {
                    "teacher_vs_baseline_margin_min_0_05": teacher_minus_baseline >= 0.05,
                    "eval_duration_under_720_seconds": float(stage_durations.get("eval", 0.0)) < 720.0,
                    "integrity_passed": integrity_passed,
                }
                run_summary["acceptance_checks"] = acceptance_checks
                if integrity_status_raw == "fail":
                    campaign_status = "needs_debug"
                    stop_reason = (
                        f"Stopped after seed {seed} due to quality integrity failure: "
                        f"{run_summary['integrity']['reason'] or 'integrity checks failed'}."
                    )
                    run_halted = True
                elif integrity_status_raw == "warn":
                    campaign_status = "needs_debug"

            run_summaries.append(run_summary)

            if run_halted:
                break

            complete_metric_runs = [run for run in run_summaries if "metrics" in run]
            if len(complete_metric_runs) == 2 and run_cfg.campaign.min_runs <= 2:
                if not all(bool(run.get("integrity", {}).get("passed", False)) for run in complete_metric_runs):
                    early_stop = {
                        "triggered": False,
                        "evaluated_after_runs": 2,
                        "checks": {"blocked_by_integrity": True},
                    }
                    continue
                decision = campaign_utils.should_early_stop_after_two_runs(
                    first=complete_metric_runs[0]["metrics"],
                    second=complete_metric_runs[1]["metrics"],
                    threshold=run_cfg.campaign.early_stop_threshold,
                )
                early_stop = {
                    "triggered": bool(decision["stop"]),
                    "evaluated_after_runs": 2,
                    "checks": decision["checks"],
                }
                if decision["stop"]:
                    stop_reason = "Stopped after 2 runs because early-stop variance criteria were satisfied."
                    break
    except Exception as exc:
        failure_class = _classify_exception(exc)
        recoverable = bool(failure_class in {"stalled", "transient_exhausted", "invariant_failed"})
        if campaign_status == "ok":
            campaign_status = "failed" if failure_class != "integrity_failed" else "needs_debug"
        if not stop_reason:
            stop_reason = str(exc)
        _emit_log(
            "command_failure",
            command="campaign",
            failure_class=failure_class,
            stop_reason=stop_reason,
            recoverable=recoverable,
        )

    complete_metrics = [run["metrics"] for run in run_summaries if "metrics" in run]
    if complete_metrics:
        across_runs = campaign_utils.summarize_across_runs(complete_metrics)
        pooled = campaign_utils.summarize_pooled_rows(
            eval_rows_by_run=run_rows,
            bootstrap_reps=cfg.campaign.bootstrap_reps,
            rng_seed=cfg.seed,
        )
    else:
        across_runs = {"runs": 0, "mean": {}, "std": {}}
        pooled = {"rows": 0, "means": {}, "ci95": {}}

    runs_with_acceptance = [run for run in run_summaries if isinstance(run.get("acceptance_checks"), dict)]
    runs_with_integrity = [run for run in run_summaries if isinstance(run.get("integrity"), dict)]
    teacher_margin_all_runs = bool(runs_with_acceptance) and all(
        bool(run.get("acceptance_checks", {}).get("teacher_vs_baseline_margin_min_0_05"))
        for run in runs_with_acceptance
    )
    eval_runtime_all_runs = bool(runs_with_acceptance) and all(
        bool(run.get("acceptance_checks", {}).get("eval_duration_under_720_seconds"))
        for run in runs_with_acceptance
    )
    integrity_all_runs = bool(runs_with_integrity) and all(
        bool(run.get("integrity", {}).get("passed", False))
        for run in runs_with_integrity
    )
    quality_checks = {
        "teacher_vs_baseline_margin_min_0_05_all_runs": teacher_margin_all_runs,
        "eval_duration_under_720_seconds_all_runs": eval_runtime_all_runs,
        "integrity_passed_all_runs": integrity_all_runs,
    }
    campaign_win = teacher_margin_all_runs and eval_runtime_all_runs and integrity_all_runs
    if campaign_status == "ok" and not campaign_win:
        campaign_status = "needs_debug"
        if not stop_reason:
            stop_reason = "Campaign completed but did not meet all win thresholds."

    reproducibility_input = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "config_fingerprint": config_fingerprint,
        "frozen_prompts_sha256": str(frozen_info_payload.get("sha256", "")),
        "planned_seeds": list(planned_seeds),
        "acceptance_thresholds": {
            "teacher_vs_baseline_margin_min": 0.05,
            "eval_duration_max_seconds_exclusive": 720.0,
            "integrity_required": True,
        },
        "run_cap": {
            "configured": strict_run_cap,
            "enforced": strict_run_cap_enforced,
            "effective_max_runs": effective_max_runs,
            "planned_max_runs": cfg.campaign.max_runs,
            "cap_hit": strict_run_cap_hit,
        },
    }
    reproducibility_artifact = {
        "campaign_status": campaign_status,
        "campaign_win": campaign_win,
        "executed_seeds": [run["seed"] for run in run_summaries],
        "quality_checks": quality_checks,
        "runs": [_minimal_campaign_run_fingerprint(run) for run in run_summaries],
    }
    reproducibility_invariant = {
        "config_fingerprint": config_fingerprint,
        "frozen_prompts_sha256": str(frozen_info_payload.get("sha256", "")),
        "run_cap": reproducibility_input["run_cap"],
        "planned_seeds": list(planned_seeds),
        "executed_seeds": [run["seed"] for run in run_summaries],
        "runs": [_campaign_run_invariant(run) for run in run_summaries],
    }
    reproducibility = {
        "mode": mode,
        "contract": (
            "deterministic_artifact_fingerprint"
            if mode == "mock"
            else "procedure_level_inputs_and_audit_invariants"
        ),
        "invariant_contract_version": "v1",
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_input)),
        "artifact_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_artifact)),
        "invariant_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_invariant)),
        "input": reproducibility_input,
        "invariant_payload": reproducibility_invariant,
    }

    report_path = campaign_dir / "campaign_report.md"
    summary_path = campaign_dir / "campaign_summary.json"
    _enforce_reproducibility(
        command="campaign",
        mode=mode,
        summary_path=summary_path,
        reproducibility_payload=reproducibility,
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "campaign_status": campaign_status,
        "campaign_win": campaign_win,
        "planned_seeds": planned_seeds,
        "executed_seeds": [run["seed"] for run in run_summaries],
        "frozen_prompts": frozen_info_payload,
        "runs": run_summaries,
        "aggregate": {
            "across_runs": across_runs,
            "pooled": pooled,
        },
        "early_stop": early_stop,
        "acceptance_checks": quality_checks,
        "guardrails": {
            "budget_caps_enforced": True,
            "projection_warning_band_enforced": False,
            "campaign_strict_run_cap": reproducibility_input["run_cap"],
        },
        "reproducibility": reproducibility,
        "spend": {
            "new_spend_usd": round(new_spend_usd, 4),
            "total_spend_usd": round(new_spend_usd, 4),
        },
        "failure_class": failure_class,
        "stop_reason": stop_reason,
        "last_completed_stage": last_completed_stage,
        "recoverable": recoverable,
    }
    summary["artifacts"] = {"campaign_summary": str(summary_path), "campaign_report": str(report_path)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(campaign_utils.format_campaign_report(summary))
    run_state["status"] = campaign_status
    run_state["stop_reason"] = stop_reason
    run_state["failure_class"] = failure_class
    _write_run_state(run_state_path, run_state)
    _emit_log(
        "command_summary",
        command="campaign",
        status=campaign_status,
        executed_runs=len(run_summaries),
        planned_runs=len(planned_seeds),
        summary_path=str(summary_path),
        report_path=str(report_path),
    )
    if campaign_status == "failed":
        raise RuntimeError(stop_reason or "Campaign failed.")
    return summary


def run_tune(
    *,
    cfg: ProjectConfig,
    mode: str,
    state_dir: Path,
    options: ExecutionOptions,
) -> dict[str, object]:
    tuning_dir = state_dir / "tuning"
    run_state_path = tuning_dir / "run_state.json"
    run_state = _read_run_state(run_state_path)
    summary_path = tuning_dir / "tuning_summary.json"
    report_path = tuning_dir / "tuning_report.md"
    candidates_path = tuning_dir / "candidates.jsonl"

    sweep_spend_usd = 0.0
    confirm_spend_usd = 0.0
    project_new_spend_usd = 0.0
    stop_reason = ""
    status = "ok"
    failure_class = ""
    last_completed_stage = ""
    recoverable = False
    stage2_frozen: tuning_utils.FrozenPromptSlice | None = None
    stage1_slices: list[tuning_utils.FrozenPromptSlice] = []
    stage1_seeds: tuple[int, ...] = tuple(cfg.campaign.seeds[:2]) or (cfg.seed,)

    candidate_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    distill_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    promoted_rows: list[dict[str, Any]] = []
    final_campaign_summary: dict[str, Any] | None = None
    final_campaign_info: dict[str, object] = {"executed": False}
    sweep_invocations = 0
    config_fingerprint = _config_fingerprint(cfg)
    strict_run_cap = int(cfg.tuning.strict_run_cap)
    strict_run_cap_enforced = strict_run_cap > 0
    phase_cap_counts: dict[str, dict[str, Any]] = {
        "teacher_headroom": {"configured": strict_run_cap, "planned": 0, "executed": 0, "cap_hit": False},
        "distill_tuning": {"configured": strict_run_cap, "planned": 0, "executed": 0, "cap_hit": False},
        "confirm": {"configured": strict_run_cap, "planned": 0, "executed": 0, "cap_hit": False},
    }

    _emit_log(
        "command_start",
        command="tune",
        mode=mode,
        config_name=cfg.name,
        state_dir=str(state_dir),
        resume=options.resume,
        heartbeat_seconds=options.heartbeat_seconds,
        progress_timeout_seconds=options.progress_timeout_seconds,
        strict_run_cap_configured=strict_run_cap,
        strict_run_cap_enforced=strict_run_cap_enforced,
    )

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean_val = _mean(values)
        variance = sum((value - mean_val) ** 2 for value in values) / len(values)
        return variance**0.5

    def _aggregate_candidate_row(*, spec: dict[str, Any], run_rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
        if not run_rows:
            return {
                **spec,
                "phase": phase,
                "status": "no_runs",
                "aggregation_runs": 0,
                "teacher_score": -999.0,
                "baseline_score": -999.0,
                "student_score": -999.0,
                "teacher_minus_baseline": -999.0,
                "student_minus_baseline": -999.0,
                "student_minus_baseline_exact_match": -999.0,
                "student_numeric_parse_rate": -999.0,
                "eval_duration_seconds": 0.0,
                "actual_spend_usd": 0.0,
                "integrity_status": "fail",
                "integrity_reason": "No stage1 runs executed",
                "integrity_pass": False,
                "teacher_margin_pass": False,
                "student_gain_pass": False,
                "student_exact_gain_pass": False,
                "student_numeric_parse_pass": False,
                "eval_runtime_pass": False,
                "composite_pass": False,
            }
        teacher_score = _mean([float(row.get("teacher_score", 0.0)) for row in run_rows])
        baseline_score = _mean([float(row.get("baseline_score", 0.0)) for row in run_rows])
        student_score = _mean([float(row.get("student_score", 0.0)) for row in run_rows])
        student_minus_baseline = _mean([float(row.get("student_minus_baseline", 0.0)) for row in run_rows])
        student_minus_baseline_exact = _mean([float(row.get("student_minus_baseline_exact_match", 0.0)) for row in run_rows])
        student_numeric_parse_rate = _mean([float(row.get("student_numeric_parse_rate", 1.0)) for row in run_rows])
        teacher_minus_baseline = _mean([float(row.get("teacher_minus_baseline", 0.0)) for row in run_rows])
        durations = [float(row.get("eval_duration_seconds", 0.0)) for row in run_rows]
        spends = [float(row.get("actual_spend_usd", 0.0)) for row in run_rows]
        integrity_passed = all(bool(row.get("integrity_pass", False)) for row in run_rows)
        checks = tuning_utils.candidate_acceptance(
            metrics={
                "means": {
                    "baseline": baseline_score,
                    "teacher": teacher_score,
                    "student": student_score,
                    "student_minus_baseline_exact_match": student_minus_baseline_exact,
                    "student_numeric_parse_rate": student_numeric_parse_rate,
                },
                "eval_duration_seconds": _mean(durations),
            },
            integrity_passed=integrity_passed,
            min_teacher_margin=cfg.tuning.min_teacher_margin,
            min_student_gain=cfg.tuning.min_student_gain,
            min_student_exact_gain=cfg.tuning.min_student_exact_gain,
            min_student_numeric_parse=cfg.tuning.min_student_numeric_parse,
            max_eval_duration_seconds=cfg.tuning.max_eval_duration_seconds,
        )
        return {
            **spec,
            "phase": phase,
            "aggregation_runs": len(run_rows),
            "stage1_seeds": list(stage1_seeds),
            "stage1_slices": [slice_info.frozen_path for slice_info in stage1_slices],
            "teacher_score": round(teacher_score, 6),
            "baseline_score": round(baseline_score, 6),
            "student_score": round(student_score, 6),
            "teacher_minus_baseline": round(teacher_minus_baseline, 6),
            "student_minus_baseline": round(student_minus_baseline, 6),
            "student_minus_baseline_std": round(_std([float(row.get("student_minus_baseline", 0.0)) for row in run_rows]), 6),
            "student_minus_baseline_exact_match": round(student_minus_baseline_exact, 6),
            "student_numeric_parse_rate": round(student_numeric_parse_rate, 6),
            "eval_duration_seconds": round(_mean(durations), 6),
            "actual_spend_usd": round(sum(spends), 6),
            "integrity_status": "pass" if integrity_passed else "fail",
            "integrity_reason": "",
            "integrity_pass": checks["integrity_pass"],
            "teacher_margin_pass": checks["teacher_margin_pass"],
            "student_gain_pass": checks["student_gain_pass"],
            "student_exact_gain_pass": checks["student_exact_gain_pass"],
            "student_numeric_parse_pass": checks["student_numeric_parse_pass"],
            "eval_runtime_pass": checks["eval_runtime_pass"],
            "composite_pass": checks["composite_pass"],
            "artifacts": {"runs": [dict(row.get("artifacts", {})) for row in run_rows]},
        }

    teacher_winner: dict[str, Any] | None = None
    winner_row: dict[str, Any] | None = None
    try:
        stage2_frozen = tuning_utils.freeze_prompt_slice(
            source_path=cfg.evaluation.prompt_file,
            frozen_path=tuning_dir / "frozen_prompts_stage2.jsonl",
            prompt_limit=0,
        )
        stage1_slices = tuning_utils.freeze_prompt_slices(
            source_path=Path(stage2_frozen.frozen_path),
            output_dir=tuning_dir / "frozen_stage1_slices",
            slice_size=0,
            num_slices=min(3, stage2_frozen.rows),
        )
        teacher_specs = tuning_utils.teacher_headroom_candidates(
            current_teacher=cfg.teacher_model,
            stronger_teacher=cfg.teacher_model,
            max_tokens_candidates=cfg.evaluation.eval_max_tokens_candidates,
            teacher_candidates=cfg.tuning.teacher_candidates,
            sweep_runs=cfg.tuning.sweep_runs,
        )
        phase_cap_counts["teacher_headroom"]["planned"] = len(teacher_specs)
        if strict_run_cap_enforced and len(teacher_specs) > strict_run_cap:
            teacher_specs = teacher_specs[:strict_run_cap]
        phase_cap_counts["teacher_headroom"]["executed"] = len(teacher_specs)
        phase_cap_counts["teacher_headroom"]["cap_hit"] = bool(
            strict_run_cap_enforced and phase_cap_counts["teacher_headroom"]["planned"] > len(teacher_specs)
        )
        for spec in teacher_specs:
            sweep_invocations += 1
            candidate_id = str(spec["candidate_id"])
            _emit_log("run_start", command="tune", run=candidate_id, phase="teacher_headroom", sweep_index=sweep_invocations)
            run_rows: list[dict[str, Any]] = []
            candidate_error = ""
            for seed in stage1_seeds:
                for slice_index, slice_info in enumerate(stage1_slices, start=1):
                    run_dir = tuning_dir / "sweeps" / "teacher" / candidate_id / f"seed-{seed}" / f"slice-{slice_index:02d}"
                    try:
                        result = _run_tune_candidate(
                            cfg=cfg,
                            mode=mode,
                            run_dir=run_dir,
                            seed=seed,
                            prompt_file=Path(slice_info.frozen_path),
                            prompt_limit=int(slice_info.rows),
                            teacher_model=str(spec["teacher_model"]),
                            max_tokens_eval=int(spec["max_tokens_eval"]),
                            eval_temperature=float(spec["eval_temperature"]),
                            teacher_prompt_template=str(spec["teacher_prompt_template"]),
                            distill_overrides={},
                            options=options,
                            run_state_path=run_state_path,
                            run_label=f"teacher:{candidate_id}:seed-{seed}:slice-{slice_index:02d}",
                            resume=options.resume,
                        )
                    except Exception as exc:
                        candidate_error = str(exc)
                        break
                    run_summary = result["run_summary"]
                    if run_summary.get("stages_completed"):
                        last_completed_stage = str(run_summary["stages_completed"][-1])
                    sweep_spend_usd = round(sweep_spend_usd + float(result["new_spend_usd"]), 4)
                    project_new_spend_usd = round(project_new_spend_usd + float(result["new_spend_usd"]), 4)
                    run_rows.append(
                        _build_tune_candidate_row(
                            spec={**spec, "stage1_seed": seed, "stage1_slice": slice_index},
                            run_summary=run_summary,
                            phase="teacher_headroom",
                            tuning_cfg=cfg.tuning,
                        )
                    )
                if candidate_error:
                    break
            if candidate_error:
                row = {
                    **spec,
                    "phase": "teacher_headroom",
                    "status": "teacher_unavailable",
                    "teacher_unavailable": True,
                    "error": candidate_error,
                    "aggregation_runs": len(run_rows),
                    "teacher_score": -999.0,
                    "baseline_score": -999.0,
                    "student_score": -999.0,
                    "teacher_minus_baseline": -999.0,
                    "student_minus_baseline": -999.0,
                    "student_minus_baseline_exact_match": -999.0,
                    "student_numeric_parse_rate": -999.0,
                    "eval_duration_seconds": 0.0,
                    "actual_spend_usd": 0.0,
                    "integrity_status": "fail",
                    "integrity_reason": candidate_error,
                    "integrity_pass": False,
                    "teacher_margin_pass": False,
                    "student_gain_pass": False,
                    "student_exact_gain_pass": False,
                    "student_numeric_parse_pass": False,
                    "eval_runtime_pass": False,
                    "composite_pass": False,
                }
            else:
                row = _aggregate_candidate_row(spec=spec, run_rows=run_rows, phase="teacher_headroom")
            candidate_rows.append(row)
            teacher_rows.append(row)
            _emit_log("run_done", command="tune", run=candidate_id, phase="teacher_headroom", integrity=row.get("integrity_pass", False))

        teacher_ranked = tuning_utils.rank_teacher_candidates(teacher_rows)
        teacher_winner = teacher_ranked[0] if teacher_ranked else None
        if teacher_winner is None and status == "ok":
            status = "needs_debug"
            stop_reason = "No teacher candidate passed integrity checks in phase-1 sweep."

        if status == "ok" and teacher_winner is not None:
            distill_specs = tuning_utils.distill_l8_candidates()
            if cfg.tuning.sweep_runs > 0:
                distill_specs = distill_specs[: cfg.tuning.sweep_runs]
            phase_cap_counts["distill_tuning"]["planned"] = len(distill_specs)
            if strict_run_cap_enforced and len(distill_specs) > strict_run_cap:
                distill_specs = distill_specs[:strict_run_cap]
            phase_cap_counts["distill_tuning"]["executed"] = len(distill_specs)
            phase_cap_counts["distill_tuning"]["cap_hit"] = bool(
                strict_run_cap_enforced and phase_cap_counts["distill_tuning"]["planned"] > len(distill_specs)
            )
            for spec in distill_specs:
                sweep_invocations += 1
                candidate_id = str(spec["candidate_id"])
                _emit_log("run_start", command="tune", run=candidate_id, phase="distill_tuning", sweep_index=sweep_invocations)
                run_rows: list[dict[str, Any]] = []
                for seed in stage1_seeds:
                    for slice_index, slice_info in enumerate(stage1_slices, start=1):
                        run_dir = tuning_dir / "sweeps" / "distill" / candidate_id / f"seed-{seed}" / f"slice-{slice_index:02d}"
                        result = _run_tune_candidate(
                            cfg=cfg,
                            mode=mode,
                            run_dir=run_dir,
                            seed=seed,
                            prompt_file=Path(slice_info.frozen_path),
                            prompt_limit=int(slice_info.rows),
                            teacher_model=str(teacher_winner["teacher_model"]),
                            max_tokens_eval=int(teacher_winner["max_tokens_eval"]),
                            eval_temperature=float(teacher_winner["eval_temperature"]),
                            teacher_prompt_template=str(teacher_winner["teacher_prompt_template"]),
                            distill_overrides={
                                "filter_profile": str(spec["filter_profile"]),
                                "hard_example_ratio": float(spec["hard_example_ratio"]),
                                "kd_alpha": float(spec["kd_alpha"]),
                                "kd_temperature": float(spec["kd_temperature"]),
                                "learning_rate": float(spec["learning_rate"]),
                                "epochs": int(spec["epochs"]),
                                "lora_rank": int(spec["lora_rank"]),
                            },
                            options=options,
                            run_state_path=run_state_path,
                            run_label=f"distill:{candidate_id}:seed-{seed}:slice-{slice_index:02d}",
                            resume=options.resume,
                        )
                        run_summary = result["run_summary"]
                        if run_summary.get("stages_completed"):
                            last_completed_stage = str(run_summary["stages_completed"][-1])
                        sweep_spend_usd = round(sweep_spend_usd + float(result["new_spend_usd"]), 4)
                        project_new_spend_usd = round(project_new_spend_usd + float(result["new_spend_usd"]), 4)
                        run_rows.append(
                            _build_tune_candidate_row(
                                spec={**spec, "stage1_seed": seed, "stage1_slice": slice_index},
                                run_summary=run_summary,
                                phase="distill_tuning",
                                tuning_cfg=cfg.tuning,
                            )
                        )
                merged = {
                    **spec,
                    "teacher_model": teacher_winner["teacher_model"],
                    "teacher_prompt_template": teacher_winner["teacher_prompt_template"],
                    "eval_temperature": teacher_winner["eval_temperature"],
                    "max_tokens_eval": teacher_winner["max_tokens_eval"],
                }
                row = _aggregate_candidate_row(spec=merged, run_rows=run_rows, phase="distill_tuning")
                candidate_rows.append(row)
                distill_rows.append(row)
                _emit_log("run_done", command="tune", run=candidate_id, phase="distill_tuning", integrity=row.get("integrity_pass", False))

        if status == "ok":
            promoted_rows = tuning_utils.promote_candidates(distill_rows, top_k=cfg.tuning.promotion_top_k)
            if not promoted_rows:
                status = "needs_debug"
                stop_reason = "No distill candidates met strict promotion gates."

        if status == "ok" and stage2_frozen is not None:
            confirm_specs = promoted_rows
            phase_cap_counts["confirm"]["planned"] = len(confirm_specs)
            if strict_run_cap_enforced and len(confirm_specs) > strict_run_cap:
                confirm_specs = confirm_specs[:strict_run_cap]
            phase_cap_counts["confirm"]["executed"] = len(confirm_specs)
            phase_cap_counts["confirm"]["cap_hit"] = bool(
                strict_run_cap_enforced and phase_cap_counts["confirm"]["planned"] > len(confirm_specs)
            )
            for promoted in confirm_specs:
                candidate_id = str(promoted["candidate_id"])
                _emit_log("run_start", command="tune", run=candidate_id, phase="confirm")
                run_dir = tuning_dir / "confirm" / candidate_id
                result = _run_tune_candidate(
                    cfg=cfg,
                    mode=mode,
                    run_dir=run_dir,
                    seed=cfg.campaign.seeds[0],
                    prompt_file=Path(stage2_frozen.frozen_path),
                    prompt_limit=int(stage2_frozen.rows),
                    teacher_model=str(promoted["teacher_model"]),
                    max_tokens_eval=int(promoted["max_tokens_eval"]),
                    eval_temperature=float(promoted["eval_temperature"]),
                    teacher_prompt_template=str(promoted["teacher_prompt_template"]),
                    distill_overrides={
                        "filter_profile": str(promoted["filter_profile"]),
                        "hard_example_ratio": float(promoted["hard_example_ratio"]),
                        "kd_alpha": float(promoted["kd_alpha"]),
                        "kd_temperature": float(promoted["kd_temperature"]),
                        "learning_rate": float(promoted["learning_rate"]),
                        "epochs": int(promoted["epochs"]),
                        "lora_rank": int(promoted["lora_rank"]),
                    },
                    options=options,
                    run_state_path=run_state_path,
                    run_label=f"confirm:{candidate_id}",
                    resume=options.resume,
                )
                run_summary = result["run_summary"]
                if run_summary.get("stages_completed"):
                    last_completed_stage = str(run_summary["stages_completed"][-1])
                confirm_spend_usd = round(confirm_spend_usd + float(result["new_spend_usd"]), 4)
                project_new_spend_usd = round(project_new_spend_usd + float(result["new_spend_usd"]), 4)
                row = _build_tune_candidate_row(
                    spec=promoted,
                    run_summary=run_summary,
                    phase="confirm",
                    tuning_cfg=cfg.tuning,
                )
                candidate_rows.append(row)
                confirmation_rows.append(row)
                _emit_log("run_done", command="tune", run=candidate_id, phase="confirm", integrity=row.get("integrity_pass", False))

        if status == "ok":
            ranked_confirmation = tuning_utils.promote_candidates(confirmation_rows, top_k=1)
            if not ranked_confirmation:
                status = "needs_debug"
                stop_reason = "No confirmation candidate passed strict gates on 150-prompt slice."
            else:
                winner_row = ranked_confirmation[0]
                if not bool(winner_row.get("composite_pass", False)):
                    status = "needs_debug"
                    stop_reason = "Top confirmation candidate failed composite acceptance checks."

        if status == "ok" and winner_row is not None and stage2_frozen is not None:
            final_eval = replace(
                cfg.evaluation,
                prompt_file=Path(stage2_frozen.frozen_path),
                prompt_limit=int(stage2_frozen.rows),
                max_tokens_eval=int(winner_row["max_tokens_eval"]),
                eval_temperature=float(winner_row["eval_temperature"]),
            )
            final_distill = replace(
                cfg.distillation,
                teacher_prompt_template=str(winner_row["teacher_prompt_template"]),
                filter_profile=str(winner_row["filter_profile"]),
                hard_example_ratio=float(winner_row["hard_example_ratio"]),
                kd_alpha=float(winner_row["kd_alpha"]),
                kd_temperature=float(winner_row["kd_temperature"]),
                learning_rate=float(winner_row["learning_rate"]),
                epochs=int(winner_row["epochs"]),
                lora_rank=int(winner_row["lora_rank"]),
            )
            final_cfg = replace(
                cfg,
                teacher_model=str(winner_row["teacher_model"]),
                evaluation=final_eval,
                distillation=final_distill,
            )
            campaign_summary = run_campaign(cfg=final_cfg, mode=mode, state_dir=tuning_dir / "final_campaign", options=options)
            final_campaign_summary = campaign_summary
            final_campaign_new_spend = float(campaign_summary.get("spend", {}).get("new_spend_usd", 0.0))
            confirm_spend_usd = round(confirm_spend_usd + final_campaign_new_spend, 4)
            project_new_spend_usd = round(project_new_spend_usd + final_campaign_new_spend, 4)
            final_campaign_info = {
                "executed": True,
                "campaign_summary_path": campaign_summary.get("artifacts", {}).get("campaign_summary", ""),
                "campaign_report_path": campaign_summary.get("artifacts", {}).get("campaign_report", ""),
                "executed_seeds": campaign_summary.get("executed_seeds", []),
                "campaign_status": campaign_summary.get("campaign_status", ""),
            }
    except Exception as exc:
        failure_class = _classify_exception(exc)
        recoverable = bool(failure_class in {"stalled", "transient_exhausted", "invariant_failed"})
        if status == "ok":
            status = "failed" if failure_class != "integrity_failed" else "needs_debug"
        if not stop_reason:
            stop_reason = str(exc)
        _emit_log(
            "command_failure",
            command="tune",
            failure_class=failure_class,
            stop_reason=stop_reason,
            recoverable=recoverable,
        )

    acceptance_checks = {
        "teacher_margin_winner_pass": bool(winner_row and bool(winner_row.get("teacher_margin_pass", False))),
        "student_gain_winner_pass": bool(winner_row and bool(winner_row.get("student_gain_pass", False))),
        "student_exact_gain_winner_pass": bool(winner_row and bool(winner_row.get("student_exact_gain_pass", False))),
        "student_numeric_parse_winner_pass": bool(
            winner_row and bool(winner_row.get("student_numeric_parse_pass", False))
        ),
        "integrity_winner_pass": bool(winner_row and bool(winner_row.get("integrity_pass", False))),
        "eval_runtime_winner_pass": bool(winner_row and bool(winner_row.get("eval_runtime_pass", False))),
        "composite_winner_pass": bool(winner_row and bool(winner_row.get("composite_pass", False))),
    }
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text("\n".join(json.dumps(row) for row in candidate_rows) + ("\n" if candidate_rows else ""))

    frozen_payload: dict[str, Any] = {"stage1_slices": [], "stage2": {}}
    if stage2_frozen is not None:
        frozen_payload["stage2"] = stage2_frozen.as_dict()
    if stage1_slices:
        frozen_payload["stage1_slices"] = [slice_info.as_dict() for slice_info in stage1_slices]

    reproducibility_input = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "config_fingerprint": config_fingerprint,
        "strict_run_cap": {
            "configured": strict_run_cap,
            "enforced": strict_run_cap_enforced,
            "per_phase": phase_cap_counts,
        },
        "stage1_seeds": list(stage1_seeds),
        "frozen_stage2_sha256": (
            str(frozen_payload.get("stage2", {}).get("sha256", ""))
            if isinstance(frozen_payload.get("stage2", {}), dict)
            else ""
        ),
        "frozen_stage1_sha256": [
            str(row.get("sha256", ""))
            for row in frozen_payload.get("stage1_slices", [])
            if isinstance(row, dict)
        ],
        "acceptance_thresholds": {
            "min_teacher_margin": cfg.tuning.min_teacher_margin,
            "min_student_gain": cfg.tuning.min_student_gain,
            "min_student_exact_gain": cfg.tuning.min_student_exact_gain,
            "min_student_numeric_parse": cfg.tuning.min_student_numeric_parse,
            "max_eval_duration_seconds": cfg.tuning.max_eval_duration_seconds,
        },
    }
    reproducibility_artifact = {
        "status": status,
        "winner_candidate_id": str(winner_row.get("candidate_id", "")) if isinstance(winner_row, dict) else "",
        "acceptance_checks": acceptance_checks,
        "teacher_rows": [_minimal_tune_candidate_fingerprint(row) for row in teacher_rows],
        "distill_rows": [_minimal_tune_candidate_fingerprint(row) for row in distill_rows],
        "confirmation_rows": [_minimal_tune_candidate_fingerprint(row) for row in confirmation_rows],
        "final_campaign_status": str(final_campaign_info.get("campaign_status", "")),
    }
    reproducibility_invariant = {
        "config_fingerprint": config_fingerprint,
        "frozen_stage2_sha256": reproducibility_input["frozen_stage2_sha256"],
        "frozen_stage1_sha256": list(reproducibility_input["frozen_stage1_sha256"]),
        "strict_run_cap": reproducibility_input["strict_run_cap"],
        "stage1_seeds": list(stage1_seeds),
        "teacher_candidate_ids_executed": [str(row.get("candidate_id", "")) for row in teacher_rows],
        "distill_candidate_ids_executed": [str(row.get("candidate_id", "")) for row in distill_rows],
        "promoted_candidate_ids": [str(row.get("candidate_id", "")) for row in promoted_rows],
        "confirmation_candidate_ids_executed": [str(row.get("candidate_id", "")) for row in confirmation_rows],
        "phase_counts": {
            "teacher_headroom": int(phase_cap_counts["teacher_headroom"]["executed"]),
            "distill_tuning": int(phase_cap_counts["distill_tuning"]["executed"]),
            "confirm": int(phase_cap_counts["confirm"]["executed"]),
        },
        "final_campaign_executed": bool(final_campaign_info.get("executed", False)),
    }
    reproducibility = {
        "mode": mode,
        "contract": (
            "deterministic_artifact_fingerprint"
            if mode == "mock"
            else "procedure_level_inputs_and_audit_invariants"
        ),
        "invariant_contract_version": "v1",
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_input)),
        "artifact_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_artifact)),
        "invariant_fingerprint": _sha256_text(_stable_json_dumps(reproducibility_invariant)),
        "input": reproducibility_input,
        "invariant_payload": reproducibility_invariant,
    }
    _enforce_reproducibility(
        command="tune",
        mode=mode,
        summary_path=summary_path,
        reproducibility_payload=reproducibility,
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "stop_reason": stop_reason,
        "failure_class": failure_class,
        "last_completed_stage": last_completed_stage,
        "recoverable": recoverable,
        "frozen_prompts": frozen_payload,
        "teacher_sweep": {"runs_executed": len(teacher_rows), "winner": teacher_winner},
        "distill_sweep": {"runs_executed": len(distill_rows)},
        "promoted_candidates": promoted_rows,
        "confirmation_runs": confirmation_rows,
        "winner": winner_row,
        "final_campaign": final_campaign_info,
        "acceptance_checks": acceptance_checks,
        "guardrails": {
            "budget_caps_enforced": True,
            "projection_warning_band_enforced": False,
            "tuning_strict_run_cap": {
                "configured": strict_run_cap,
                "enforced": strict_run_cap_enforced,
                "per_phase": phase_cap_counts,
            },
        },
        "reproducibility": reproducibility,
        "execution_counts": {
            "candidates_total": len(candidate_rows),
            "teacher_candidates": len(teacher_rows),
            "distill_candidates": len(distill_rows),
            "confirmation_candidates": len(confirmation_rows),
            "sweep_invocations": sweep_invocations,
            "strict_run_cap_configured": cfg.tuning.strict_run_cap,
            "strict_run_cap_enforced": strict_run_cap_enforced,
            "strict_run_cap_per_phase": phase_cap_counts,
        },
        "spend": {
            "sweep_spend_usd": round(sweep_spend_usd, 4),
            "confirm_spend_usd": round(confirm_spend_usd, 4),
            "new_spend_usd": round(project_new_spend_usd, 4),
            "total_spend_usd": round(project_new_spend_usd, 4),
        },
        "artifacts": {
            "tuning_summary": str(summary_path),
            "tuning_report": str(report_path),
            "candidates": str(candidates_path),
        },
    }
    if final_campaign_summary is not None:
        summary["final_campaign_summary"] = final_campaign_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report_path.write_text(tuning_utils.format_tuning_report(summary))
    run_state["status"] = status
    run_state["stop_reason"] = stop_reason
    run_state["failure_class"] = failure_class
    _write_run_state(run_state_path, run_state)
    _emit_log(
        "command_summary",
        command="tune",
        status=status,
        sweep_invocations=sweep_invocations,
        strict_run_cap_configured=cfg.tuning.strict_run_cap,
        strict_run_cap_enforced=strict_run_cap_enforced,
        summary_path=str(summary_path),
        report_path=str(report_path),
    )
    if status == "failed":
        raise RuntimeError(stop_reason or "Tune failed.")
    return summary


def _build_tune_candidate_row(
    *,
    spec: dict[str, Any],
    run_summary: dict[str, Any],
    phase: str,
    tuning_cfg: Any,
) -> dict[str, Any]:
    metrics = run_summary.get("metrics", {})
    means = metrics.get("means", {}) if isinstance(metrics, dict) else {}
    teacher_minus_baseline = float(means.get("teacher", 0.0)) - float(means.get("baseline", 0.0))
    student_minus_baseline = float(means.get("student", 0.0)) - float(means.get("baseline", 0.0))
    student_minus_baseline_exact = float(means.get("student_minus_baseline_exact_match", 0.0))
    student_numeric_parse_rate = float(means.get("student_numeric_parse_rate", 1.0))
    eval_duration_seconds = float(run_summary.get("stage_durations_seconds", {}).get("eval", 0.0))
    integrity = run_summary.get("integrity", {})
    integrity_passed = bool(integrity.get("passed", False))
    checks = tuning_utils.candidate_acceptance(
        metrics={
            "means": {
                **means,
                "student_minus_baseline_exact_match": student_minus_baseline_exact,
                "student_numeric_parse_rate": student_numeric_parse_rate,
            },
            "eval_duration_seconds": eval_duration_seconds,
        },
        integrity_passed=integrity_passed,
        min_teacher_margin=float(tuning_cfg.min_teacher_margin),
        min_student_gain=float(tuning_cfg.min_student_gain),
        min_student_exact_gain=float(tuning_cfg.min_student_exact_gain),
        min_student_numeric_parse=float(tuning_cfg.min_student_numeric_parse),
        max_eval_duration_seconds=float(tuning_cfg.max_eval_duration_seconds),
    )
    row: dict[str, Any] = {
        **spec,
        "phase": phase,
        "seed": int(run_summary.get("seed", 0)),
        "run_dir": str(run_summary.get("run_dir", "")),
        "teacher_score": round(float(means.get("teacher", 0.0)), 6),
        "baseline_score": round(float(means.get("baseline", 0.0)), 6),
        "student_score": round(float(means.get("student", 0.0)), 6),
        "teacher_minus_baseline": round(teacher_minus_baseline, 6),
        "student_minus_baseline": round(student_minus_baseline, 6),
        "student_minus_baseline_exact_match": round(student_minus_baseline_exact, 6),
        "student_numeric_parse_rate": round(student_numeric_parse_rate, 6),
        "eval_duration_seconds": eval_duration_seconds,
        "actual_spend_usd": float(run_summary.get("actual_spend_usd", 0.0)),
        "integrity_status": str(integrity.get("status", "")),
        "integrity_reason": str(integrity.get("reason", "")),
        "integrity_pass": checks["integrity_pass"],
        "teacher_margin_pass": checks["teacher_margin_pass"],
        "student_gain_pass": checks["student_gain_pass"],
        "student_exact_gain_pass": checks["student_exact_gain_pass"],
        "student_numeric_parse_pass": checks["student_numeric_parse_pass"],
        "eval_runtime_pass": checks["eval_runtime_pass"],
        "composite_pass": checks["composite_pass"],
        "artifacts": dict(run_summary.get("artifacts", {})),
    }
    return row


def _run_tune_candidate(
    *,
    cfg: ProjectConfig,
    mode: str,
    run_dir: Path,
    seed: int,
    prompt_file: Path,
    prompt_limit: int,
    teacher_model: str,
    max_tokens_eval: int,
    eval_temperature: float,
    teacher_prompt_template: str,
    distill_overrides: dict[str, object],
    options: ExecutionOptions,
    run_state_path: Path,
    run_label: str,
    resume: bool,
) -> dict[str, object]:
    run_paths = PipelinePaths(run_dir)
    run_cfg = replace(
        cfg,
        seed=seed,
        teacher_model=teacher_model,
        evaluation=replace(
            cfg.evaluation,
            prompt_file=prompt_file,
            prompt_limit=prompt_limit,
            max_tokens_eval=max_tokens_eval,
            eval_temperature=eval_temperature,
        ),
        distillation=replace(
            cfg.distillation,
            training_prompt_file=prompt_file,
            teacher_prompt_template=teacher_prompt_template,
            **distill_overrides,
        ),
    )
    preflight_result = ensure_preflight_ready(mode=mode, cfg=run_cfg, state_dir=run_dir)
    adapters = select_stage_adapters(mode)

    stage_durations: dict[str, float] = {}
    completed_stages: list[str] = []
    candidate_new_spend = 0.0
    run_state = _read_run_state(run_state_path)
    attempts = max(1, int(cfg.runtime.max_consecutive_failures))

    for stage in (*REQUIRED_STAGES, "report"):
        if resume and _is_stage_completed(run_state, run_id=run_label, stage=stage):
            _verify_resumed_stage(stage=stage, paths=run_paths)
            completed_stages.append(stage)
            _emit_log("stage_resume_skip", command="tune", run=run_label, stage=stage)
            continue

        for attempt in range(1, attempts + 1):
            stage_started = time.monotonic()
            ledger_before = load_ledger(run_paths.ledger)
            records_before = len(ledger_before.records)
            _emit_log(
                "stage_start",
                command="tune",
                run=run_label,
                stage=stage,
                attempt=attempt,
                max_attempts=attempts,
            )

            def _stage_call() -> None:
                if stage == "rl":
                    run_rl(run_cfg, run_paths, adapter=adapters.rl, mode=mode, preflight=preflight_result)
                elif stage == "teacher_ft":
                    run_teacher_ft(run_cfg, run_paths, adapter=adapters.teacher_ft, mode=mode, preflight=preflight_result)
                elif stage == "distill":
                    run_distill(run_cfg, run_paths, adapter=adapters.distill, mode=mode, preflight=preflight_result)
                elif stage == "eval":
                    run_eval(run_cfg, run_paths, adapter=adapters.eval, mode=mode, preflight=preflight_result)
                elif stage == "report":
                    run_report(run_cfg, run_paths, mode=mode, preflight=preflight_result)

            try:
                _run_with_watchdog(
                    fn=_stage_call,
                    context=StageExecutionContext(command="tune", run_id=run_label, stage=stage),
                    options=options,
                )
                stage_durations[stage] = round(time.monotonic() - stage_started, 4)
                ledger_after = _verify_campaign_stage(stage=stage, paths=run_paths, ledger_records_before=records_before)
                completed_stages.append(stage)
                _mark_completed_stage(run_state, run_id=run_label, stage=stage)
                _write_run_state(run_state_path, run_state)
                spend_delta = 0.0
                if stage in REQUIRED_STAGES:
                    delta = round(ledger_after.total_spend_usd - ledger_before.total_spend_usd, 4)
                    if delta < 0:
                        raise RuntimeError(f"Tune candidate stage '{stage}' produced negative spend delta: {delta}")
                    candidate_new_spend = round(candidate_new_spend + delta, 4)
                    spend_delta = delta
                _emit_log(
                    "stage_done",
                    command="tune",
                    run=run_label,
                    stage=stage,
                    duration_seconds=stage_durations[stage],
                    spend_delta_usd=round(spend_delta, 4),
                )
                break
            except Exception as exc:
                if attempt >= attempts:
                    raise StageExecutionError(
                        f"Tune stage failed after retries: run={run_label} stage={stage} error={exc}",
                        failure_class=_classify_exception(exc),
                    ) from exc
                _emit_log(
                    "stage_retry",
                    command="tune",
                    run=run_label,
                    stage=stage,
                    attempt=attempt,
                    error=str(exc),
                )
                sleep_seconds = min(
                    cfg.runtime.retry_delay_max_seconds,
                    cfg.runtime.retry_delay_base_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(sleep_seconds)

    run_summary: dict[str, Any] = {
        "seed": run_cfg.seed,
        "run_dir": str(run_dir),
        "stages_completed": completed_stages,
        "stage_durations_seconds": stage_durations,
        "actual_spend_usd": round(load_ledger(run_paths.ledger).total_spend_usd, 4),
        "stop_reason": "",
        "artifacts": {
            "eval_report": str(run_paths.report_md),
            "run_audit_report": str(run_paths.audit_report_md),
            "eval_rows": str(run_paths.eval_rows),
            "ledger": str(run_paths.ledger),
        },
    }
    if "eval" in completed_stages:
        eval_rows = audit.load_eval_rows(run_paths.eval_rows)
        if not eval_rows:
            raise RuntimeError(f"Tune candidate did not emit eval rows: {run_paths.eval_rows}")
        run_summary["metrics"] = campaign_utils.summarize_eval_rows(
            eval_rows=eval_rows,
            bootstrap_reps=run_cfg.campaign.bootstrap_reps,
            rng_seed=run_cfg.seed,
        )
        eval_metrics = _read_json(run_paths.eval_metrics)
        integrity = eval_metrics.get("integrity", {})
        integrity_status_raw = str(integrity.get("status", "pass"))
        if integrity_status_raw == "ok":
            integrity_status_raw = "pass"
        if integrity_status_raw == "integrity_failed":
            integrity_status_raw = "fail"
        integrity_passed = bool(integrity.get("passed", integrity_status_raw == "pass")) and integrity_status_raw == "pass"
        run_summary["integrity"] = {
            "passed": integrity_passed,
            "status": integrity_status_raw,
            "reason": str(integrity.get("reason", "")),
            "checks": dict(integrity.get("checks", {})) if isinstance(integrity.get("checks"), dict) else {},
        }

    return {
        "run_summary": run_summary,
        "new_spend_usd": round(candidate_new_spend, 4),
    }


def _projected_and_actual_mock(stage: str, cfg: ProjectConfig) -> tuple[TokenUsage, float, TokenUsage, float]:
    projected_tokens = budget.stage_token_usage(stage, cfg)
    projected_cost = budget.projected_stage_cost_usd(stage, cfg)
    # Keep actual spend under projection for predictable low-cost demo behavior.
    actual_factor = {"rl": 0.94, "teacher_ft": 0.96, "distill": 0.93, "eval": 0.90}[stage]
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
    return ledger, projected_tokens, projected_cost


def _enforce_stage_budget_caps(
    *,
    stage: str,
    projected_tokens: TokenUsage,
    projected_cost: float,
    actual_tokens: TokenUsage,
    actual_cost: float,
) -> None:
    token_violations: list[str] = []
    if actual_tokens.prefill > projected_tokens.prefill:
        token_violations.append(f"prefill={actual_tokens.prefill} > cap={projected_tokens.prefill}")
    if actual_tokens.sample > projected_tokens.sample:
        token_violations.append(f"sample={actual_tokens.sample} > cap={projected_tokens.sample}")
    if actual_tokens.train > projected_tokens.train:
        token_violations.append(f"train={actual_tokens.train} > cap={projected_tokens.train}")

    cost_violation = actual_cost > (projected_cost + 1e-6)
    if token_violations or cost_violation:
        details: list[str] = []
        if token_violations:
            details.append("token caps violated (" + ", ".join(token_violations) + ")")
        if cost_violation:
            details.append(f"cost={round(actual_cost, 6)} > cap={round(projected_cost, 6)}")
        raise GuardrailViolationError(
            f"Stage '{stage}' exceeded configured budget cap: " + "; ".join(details),
            failure_class="budget_cap_exceeded",
        )


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
    _enforce_stage_budget_caps(
        stage=stage,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
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


def run_teacher_ft(
    cfg: ProjectConfig,
    paths: PipelinePaths,
    *,
    adapter: TeacherFTStageAdapter,
    mode: str,
    preflight: PreflightResult,
) -> None:
    stage = "teacher_ft"
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
    _enforce_stage_budget_caps(
        stage=stage,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
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
    _enforce_stage_budget_caps(
        stage=stage,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
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
    _enforce_stage_budget_caps(
        stage=stage,
        projected_tokens=projected_tokens,
        projected_cost=projected_cost,
        actual_tokens=actual_tokens,
        actual_cost=actual_cost,
    )
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
            f"- Teacher FT stability score: {stability['teacher_ft']['stability_score']} (NaN events: {stability['teacher_ft']['nan_events']})",
            f"- Distill stability score: {stability['distill']['stability_score']} (NaN events: {stability['distill']['nan_events']})",
            "",
            "## Stage Spend",
        ]
    )
    for stage in ("rl", "teacher_ft", "distill", "eval"):
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
