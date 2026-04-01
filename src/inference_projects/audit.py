from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from inference_projects.config import ProjectConfig, REQUIRED_STAGES
from inference_projects.schemas import (
    SCHEMA_VERSION,
    validate_audit_eval_row,
    validate_run_manifest_payload,
    validate_stage_audit_payload,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    validate_run_manifest_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def upsert_run_manifest(
    *,
    path: Path,
    cfg: ProjectConfig,
    mode: str,
    state_dir: Path,
    stage: str,
    started_at: str,
    finished_at: str,
    warnings: list[str],
) -> None:
    manifest = load_json(path)
    stage_order = [*REQUIRED_STAGES, "report"]
    existing = manifest.get("stages_completed", [])
    completed = [name for name in stage_order if name in set(existing + [stage])]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "project_name": cfg.name,
        "started_at": str(manifest.get("started_at", started_at)),
        "updated_at": finished_at,
        "stages_expected": stage_order,
        "stages_completed": completed,
        "target_cap_usd": cfg.budget.target_cap_usd,
        "hard_cap_usd": cfg.budget.hard_cap_usd,
        "state_dir": str(state_dir),
        "warnings": list(warnings),
    }
    write_run_manifest(path, payload)


def write_stage_audit(path: Path, payload: dict[str, Any]) -> None:
    validate_stage_audit_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_eval_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        validate_audit_eval_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def load_eval_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def format_run_audit_report(
    *,
    cfg: ProjectConfig,
    mode: str,
    manifest: dict[str, Any],
    stage_payloads: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {cfg.name} Run Audit Report",
        "",
        "## Run Metadata",
        f"- Run mode: {mode}",
        f"- Started at: {manifest.get('started_at', 'n/a')}",
        f"- Last updated: {manifest.get('updated_at', 'n/a')}",
        f"- State dir: {manifest.get('state_dir', 'n/a')}",
        f"- Stages completed: {', '.join(manifest.get('stages_completed', [])) or 'none'}",
        "",
        "## Stage Chronology",
    ]

    if not stage_payloads:
        lines.append("- No stage audit payloads were found.")
    for payload in stage_payloads:
        stage = payload.get("stage", "unknown")
        lines.extend(
            [
                f"### Stage `{stage}`",
                f"- Status: {payload.get('status', 'unknown')}",
                f"- Started at: {payload.get('started_at', 'n/a')}",
                f"- Finished at: {payload.get('finished_at', 'n/a')}",
                f"- Duration (s): {payload.get('duration_seconds', 'n/a')}",
                f"- Projected cost (USD): {payload.get('projected_cost_usd', 'n/a')}",
                f"- Actual cost (USD): {payload.get('actual_cost_usd', 'n/a')}",
                f"- Cumulative before (USD): {payload.get('cumulative_total_before_usd', 'n/a')}",
                f"- Cumulative after (USD): {payload.get('cumulative_total_after_usd', 'n/a')}",
                f"- Usage run_id: {payload.get('usage', {}).get('run_id', 'n/a')}",
                f"- Provider raw: {json.dumps(payload.get('usage', {}).get('provider_raw', {}), sort_keys=True)}",
            ]
        )

    lines.extend(["", "## Prompt-Level Evidence Appendix"])
    if not eval_rows:
        lines.append("- No eval row traces found.")
    else:
        lines.append(f"- Total eval rows: {len(eval_rows)}")
        for row in eval_rows:
            lines.extend(
                [
                    f"### Row `{row.get('row_id', 'unknown')}`",
                    f"- Prompt: {row.get('prompt', '')}",
                    f"- Reference: {row.get('reference', '')}",
                    f"- Baseline output: {row.get('baseline_output', '')}",
                    f"- Teacher output: {row.get('teacher_output', '')}",
                    f"- Student output: {row.get('student_output', '')}",
                    f"- Scores: baseline={row.get('baseline_overlap', 0)}, teacher={row.get('teacher_overlap', 0)}, student={row.get('student_overlap', 0)}",
                    f"- Wins: vs baseline={row.get('student_vs_baseline_win', 0)}, vs teacher={row.get('student_vs_teacher_win', 0)}",
                ]
            )

    lines.extend(["", "## Anomalies and Notes"])
    warnings = manifest.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- Warning: {warning}")
    else:
        lines.append("- No anomalies recorded.")
    lines.append("")
    return "\n".join(lines)
