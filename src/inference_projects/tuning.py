from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrozenPromptSlice:
    source_path: str
    frozen_path: str
    sha256: str
    rows: int
    prompt_limit: int
    prompt_ids: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "frozen_path": self.frozen_path,
            "sha256": self.sha256,
            "rows": self.rows,
            "prompt_limit": self.prompt_limit,
            "prompt_ids": list(self.prompt_ids),
        }


def freeze_prompt_slice(*, source_path: Path, frozen_path: Path, prompt_limit: int) -> FrozenPromptSlice:
    if prompt_limit < 0:
        raise ValueError("prompt_limit must be >= 0")
    rows = _load_jsonl_rows(source_path)
    selected = rows
    prompt_ids: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(selected, start=1):
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not row_id:
            raise RuntimeError(f"Prompt row {idx} missing non-empty 'id': {source_path}")
        if not prompt:
            raise RuntimeError(f"Prompt row {idx} missing non-empty 'prompt': {source_path}")
        if row_id in seen:
            raise RuntimeError(f"Prompt rows contain duplicate id '{row_id}': {source_path}")
        seen.add(row_id)
        prompt_ids.append(row_id)

    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text("\n".join(json.dumps(row) for row in selected) + "\n")
    effective_limit = len(selected)

    return FrozenPromptSlice(
        source_path=str(source_path),
        frozen_path=str(frozen_path),
        sha256=_sha256_file(frozen_path),
        rows=effective_limit,
        prompt_limit=effective_limit,
        prompt_ids=prompt_ids,
    )


def freeze_prompt_slices(
    *,
    source_path: Path,
    output_dir: Path,
    slice_size: int,
    num_slices: int,
) -> list[FrozenPromptSlice]:
    if slice_size < 0:
        raise ValueError("slice_size must be >= 0")
    if num_slices <= 0:
        raise ValueError("num_slices must be > 0")
    _ = slice_size  # cap-like prompt limits are telemetry-only
    rows = _load_jsonl_rows(source_path)
    if not rows:
        raise RuntimeError(f"Prompt file has no rows: {source_path}")

    actual_slices = min(num_slices, len(rows))
    base_size, remainder = divmod(len(rows), actual_slices)

    output_dir.mkdir(parents=True, exist_ok=True)
    slices: list[FrozenPromptSlice] = []
    start = 0
    for idx in range(actual_slices):
        current_size = base_size + (1 if idx < remainder else 0)
        end = start + current_size
        chunk = rows[start:end]
        frozen_path = output_dir / f"slice_{idx + 1:02d}.jsonl"
        frozen_path.write_text("\n".join(json.dumps(row) for row in chunk) + "\n")
        prompt_ids = [str(row.get("id", "")).strip() for row in chunk]
        if any(not row_id for row_id in prompt_ids):
            raise RuntimeError(f"Prompt slice {idx + 1} has empty row ids: {source_path}")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise RuntimeError(f"Prompt slice {idx + 1} has duplicate row ids: {source_path}")
        slices.append(
            FrozenPromptSlice(
                source_path=str(source_path),
                frozen_path=str(frozen_path),
                sha256=_sha256_file(frozen_path),
                rows=current_size,
                prompt_limit=current_size,
                prompt_ids=prompt_ids,
            )
        )
        start = end
    return slices


def teacher_headroom_candidates(
    *,
    current_teacher: str,
    stronger_teacher: str,
    max_tokens_candidates: tuple[int, ...],
    teacher_candidates: tuple[str, ...] | None = None,
    sweep_runs: int | None = None,
) -> list[dict[str, object]]:
    token_levels = _pick_two_levels(max_tokens_candidates)
    prompt_levels = ("raw", "numeric_strict")
    temp_levels = (0.0, 0.2)

    # 2^(4-1) fractional factorial (8-run): [A, B, C, D]
    design = (
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 1),
    )
    candidates: list[dict[str, object]] = []
    if teacher_candidates:
        model_levels = tuple(dict.fromkeys(str(model).strip() for model in teacher_candidates if str(model).strip()))
        if not model_levels:
            raise ValueError("teacher_headroom_candidates requires at least one teacher model")
        candidate_index = 1
        for teacher_model in model_levels:
            for row in design:
                candidates.append(
                    {
                        "candidate_id": f"teacher-{candidate_index:02d}",
                        "phase": "teacher_headroom",
                        "teacher_model": teacher_model,
                        "teacher_prompt_template": prompt_levels[row[1]],
                        "eval_temperature": temp_levels[row[2]],
                        "max_tokens_eval": token_levels[row[3]],
                    }
                )
                candidate_index += 1
    else:
        model_levels = (current_teacher, stronger_teacher)
        for idx, row in enumerate(design, start=1):
            candidates.append(
                {
                    "candidate_id": f"teacher-{idx:02d}",
                    "phase": "teacher_headroom",
                    "teacher_model": model_levels[row[0]],
                    "teacher_prompt_template": prompt_levels[row[1]],
                    "eval_temperature": temp_levels[row[2]],
                    "max_tokens_eval": token_levels[row[3]],
                }
            )
    if sweep_runs is not None and sweep_runs > 0:
        return candidates[:sweep_runs]
    return candidates


def distill_l8_candidates() -> list[dict[str, object]]:
    # Taguchi L8 for 7 two-level factors.
    l8 = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 1),
        (0, 1, 1, 0, 0, 1, 1),
        (0, 1, 1, 1, 1, 0, 0),
        (1, 0, 1, 0, 1, 0, 1),
        (1, 0, 1, 1, 0, 1, 0),
        (1, 1, 0, 0, 1, 1, 0),
        (1, 1, 0, 1, 0, 0, 1),
    )
    low_high = {
        "filter_profile": ("moderate", "strict"),
        "hard_example_ratio": (0.4, 0.7),
        "kd_alpha": (0.3, 0.7),
        "kd_temperature": (1.0, 4.0),
        "learning_rate": (0.00001, 0.00005),
        "epochs": (1, 3),
        "lora_rank": (8, 16),
    }
    keys = tuple(low_high.keys())
    candidates: list[dict[str, object]] = []
    for idx, row in enumerate(l8, start=1):
        candidate: dict[str, object] = {
            "candidate_id": f"distill-{idx:02d}",
            "phase": "distill_tuning",
        }
        for key, level in zip(keys, row):
            candidate[key] = low_high[key][level]
        candidates.append(candidate)
    return candidates


def candidate_acceptance(
    *,
    metrics: dict[str, Any],
    integrity_passed: bool,
    min_teacher_margin: float = 0.05,
    min_student_gain: float = 0.03,
    min_student_exact_gain: float = 0.02,
    min_student_numeric_parse: float = 0.95,
    max_eval_duration_seconds: float = 720.0,
) -> dict[str, bool]:
    means = metrics.get("means", {}) if isinstance(metrics, dict) else {}
    teacher_margin = float(means.get("teacher", 0.0)) - float(means.get("baseline", 0.0))
    student_gain = float(means.get("student", 0.0)) - float(means.get("baseline", 0.0))
    student_exact_gain = float(means.get("student_minus_baseline_exact_match", 0.0))
    student_numeric_parse = float(means.get("student_numeric_parse_rate", 1.0))
    eval_duration_seconds = float(metrics.get("eval_duration_seconds", 0.0))
    teacher_margin_pass = teacher_margin >= min_teacher_margin
    student_gain_pass = student_gain >= min_student_gain
    student_exact_gain_pass = student_exact_gain >= min_student_exact_gain
    student_numeric_parse_pass = student_numeric_parse >= min_student_numeric_parse
    eval_runtime_pass = eval_duration_seconds <= max_eval_duration_seconds
    integrity_pass = bool(integrity_passed)
    composite_pass = (
        teacher_margin_pass
        and student_gain_pass
        and student_exact_gain_pass
        and student_numeric_parse_pass
        and eval_runtime_pass
        and integrity_pass
    )
    return {
        "teacher_margin_pass": teacher_margin_pass,
        "student_gain_pass": student_gain_pass,
        "student_exact_gain_pass": student_exact_gain_pass,
        "student_numeric_parse_pass": student_numeric_parse_pass,
        "eval_runtime_pass": eval_runtime_pass,
        "integrity_pass": integrity_pass,
        "composite_pass": composite_pass,
    }


def rank_teacher_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [row for row in candidates if bool(row.get("integrity_pass", False))]
    return sorted(
        filtered,
        key=lambda row: (
            float(row.get("teacher_minus_baseline", -999.0)),
            float(row.get("teacher_score", -999.0)),
        ),
        reverse=True,
    )


def promote_candidates(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    passing = [
        row
        for row in candidates
        if bool(row.get("teacher_margin_pass"))
        and bool(row.get("student_gain_pass"))
        and bool(row.get("student_exact_gain_pass", False))
        and bool(row.get("student_numeric_parse_pass", False))
        and bool(row.get("eval_runtime_pass", False))
        and bool(row.get("integrity_pass"))
    ]
    ranked = sorted(
        passing,
        key=lambda row: (
            float(row.get("student_minus_baseline", -999.0)),
            float(row.get("student_minus_baseline_exact_match", -999.0)),
            float(row.get("student_numeric_parse_rate", -999.0)),
            -float(row.get("eval_duration_seconds", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def format_tuning_report(summary: dict[str, Any]) -> str:
    spend = summary.get("spend", {})
    checks = summary.get("acceptance_checks", {})
    winner = summary.get("winner", {})
    lines = [
        "# Tuning Report",
        "",
        "## Status",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Stop reason: {summary.get('stop_reason', '') or 'none'}",
        "",
        "## Spend Telemetry",
        f"- Sweep spend (USD): {spend.get('sweep_spend_usd', 0.0)}",
        f"- Confirm spend (USD): {spend.get('confirm_spend_usd', 0.0)}",
        f"- Total spend (USD): {spend.get('total_spend_usd', 0.0)}",
        "",
        "## Sweep",
        f"- Teacher runs executed: {summary.get('teacher_sweep', {}).get('runs_executed', 0)}",
        f"- Distill runs executed: {summary.get('distill_sweep', {}).get('runs_executed', 0)}",
        f"- Promoted candidates: {len(summary.get('promoted_candidates', []))}",
        "",
        "## Winner",
        f"- Winner candidate id: {winner.get('candidate_id', 'n/a') if isinstance(winner, dict) else 'n/a'}",
        f"- Winner student-baseline: {winner.get('student_minus_baseline', 'n/a') if isinstance(winner, dict) else 'n/a'}",
        f"- Winner teacher-baseline: {winner.get('teacher_minus_baseline', 'n/a') if isinstance(winner, dict) else 'n/a'}",
        "",
        "## Acceptance Checks",
        f"- Teacher margin >= +0.05 (winner): {checks.get('teacher_margin_winner_pass', False)}",
        f"- Student gain >= +0.03 (winner): {checks.get('student_gain_winner_pass', False)}",
        f"- Integrity passed (winner): {checks.get('integrity_winner_pass', False)}",
        f"- Eval duration < 720s (winner): {checks.get('eval_runtime_winner_pass', False)}",
        "",
        "## Confirmation",
        f"- Final campaign executed: {summary.get('final_campaign', {}).get('executed', False)}",
        f"- Campaign summary path: {summary.get('final_campaign', {}).get('campaign_summary_path', '') or 'n/a'}",
        "",
    ]
    return "\n".join(lines)


def _pick_two_levels(values: tuple[int, ...]) -> tuple[int, int]:
    unique = sorted(set(int(value) for value in values))
    if len(unique) >= 2:
        return unique[0], unique[-1]
    if len(unique) == 1:
        return unique[0], unique[0]
    raise ValueError("Need at least one max token candidate")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"Expected JSON object rows in prompt file: {path}")
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
