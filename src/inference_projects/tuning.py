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
    rows = _load_jsonl_rows(source_path)
    if prompt_limit <= 0:
        raise ValueError("prompt_limit must be > 0")
    if len(rows) < prompt_limit:
        raise RuntimeError(
            f"Prompt file has {len(rows)} rows, but required at least {prompt_limit}: {source_path}"
        )

    selected = rows[:prompt_limit]
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

    return FrozenPromptSlice(
        source_path=str(source_path),
        frozen_path=str(frozen_path),
        sha256=_sha256_file(frozen_path),
        rows=prompt_limit,
        prompt_limit=prompt_limit,
        prompt_ids=prompt_ids,
    )


def teacher_headroom_candidates(
    *,
    current_teacher: str,
    stronger_teacher: str,
    max_tokens_candidates: tuple[int, ...],
) -> list[dict[str, object]]:
    token_levels = _pick_two_levels(max_tokens_candidates)
    model_levels = (current_teacher, stronger_teacher)
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
) -> dict[str, bool]:
    means = metrics.get("means", {}) if isinstance(metrics, dict) else {}
    teacher_margin = float(means.get("teacher", 0.0)) - float(means.get("baseline", 0.0))
    student_gain = float(means.get("student", 0.0)) - float(means.get("baseline", 0.0))
    return {
        "teacher_margin_pass": teacher_margin >= 0.05,
        "student_gain_pass": student_gain >= 0.03,
        "integrity_pass": bool(integrity_passed),
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
        and bool(row.get("integrity_pass"))
    ]
    ranked = sorted(
        passing,
        key=lambda row: (
            float(row.get("student_minus_baseline", -999.0)),
            float(row.get("student_minus_baseline_exact_match", -999.0)),
            -float(row.get("eval_duration_seconds", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def format_tuning_report(summary: dict[str, Any]) -> str:
    budget = summary.get("budget", {})
    checks = summary.get("acceptance_checks", {})
    winner = summary.get("winner", {})
    lines = [
        "# Tuning Report",
        "",
        "## Status",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Stop reason: {summary.get('stop_reason', '') or 'none'}",
        "",
        "## Budget",
        f"- Hard cap (USD): {budget.get('hard_cap_usd', 0.0)}",
        f"- Prior spend (USD): {budget.get('prior_spend_usd', 0.0)}",
        f"- Sweep cap (USD): {budget.get('sweep_cap_usd', 0.0)}",
        f"- Confirm cap (USD): {budget.get('confirm_cap_usd', 0.0)}",
        f"- Sweep spend (USD): {budget.get('sweep_spend_usd', 0.0)}",
        f"- Confirm spend (USD): {budget.get('confirm_spend_usd', 0.0)}",
        f"- Total spend (USD): {budget.get('total_spend_usd', 0.0)}",
        f"- Cap hit: {budget.get('cap_hit', False)}",
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
