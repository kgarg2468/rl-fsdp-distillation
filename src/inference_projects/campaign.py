from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import fmean, stdev
from typing import Any


@dataclass(frozen=True)
class FrozenPromptInfo:
    source_path: str
    frozen_path: str
    sha256: str
    total_rows: int
    prompt_limit: int
    prompt_ids: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "frozen_path": self.frozen_path,
            "sha256": self.sha256,
            "total_rows": self.total_rows,
            "prompt_limit": self.prompt_limit,
            "prompt_ids": list(self.prompt_ids),
        }


def freeze_prompt_file(*, source_path: Path, frozen_path: Path, prompt_limit: int) -> FrozenPromptInfo:
    rows = _load_jsonl_rows(source_path)
    if len(rows) < prompt_limit:
        raise RuntimeError(
            f"Prompt file has {len(rows)} rows, but prompt_limit requires at least {prompt_limit}: {source_path}"
        )

    ids: list[str] = []
    seen: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        if not row_id:
            raise RuntimeError(f"Prompt file row {idx} missing non-empty 'id': {source_path}")
        if not prompt:
            raise RuntimeError(f"Prompt file row {idx} missing non-empty 'prompt': {source_path}")
        if row_id in seen:
            raise RuntimeError(f"Prompt file contains duplicate id '{row_id}': {source_path}")
        seen.add(row_id)
        ids.append(row_id)

    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(source_path.read_text())
    digest = _sha256_file(frozen_path)
    return FrozenPromptInfo(
        source_path=str(source_path),
        frozen_path=str(frozen_path),
        sha256=digest,
        total_rows=len(rows),
        prompt_limit=prompt_limit,
        prompt_ids=ids[:prompt_limit],
    )


def summarize_eval_rows(
    *,
    eval_rows: list[dict[str, Any]],
    bootstrap_reps: int,
    rng_seed: int,
) -> dict[str, object]:
    baseline = [float(row["baseline_overlap"]) for row in eval_rows]
    teacher = [float(row["teacher_overlap"]) for row in eval_rows]
    student = [float(row["student_overlap"]) for row in eval_rows]
    delta = [s - b for s, b in zip(student, baseline)]
    ratio = [_safe_ratio(s, t) for s, t in zip(student, teacher)]

    return {
        "rows": len(eval_rows),
        "means": {
            "baseline": round(_mean(baseline), 6),
            "teacher": round(_mean(teacher), 6),
            "student": round(_mean(student), 6),
            "student_minus_baseline": round(_mean(delta), 6),
            "student_teacher_ratio": round(_mean(ratio), 6),
        },
        "ci95": {
            "baseline": _round_pair(bootstrap_mean_ci(baseline, reps=bootstrap_reps, rng_seed=rng_seed + 11)),
            "teacher": _round_pair(bootstrap_mean_ci(teacher, reps=bootstrap_reps, rng_seed=rng_seed + 17)),
            "student": _round_pair(bootstrap_mean_ci(student, reps=bootstrap_reps, rng_seed=rng_seed + 23)),
            "student_minus_baseline": _round_pair(
                bootstrap_mean_ci(delta, reps=bootstrap_reps, rng_seed=rng_seed + 31)
            ),
            "student_teacher_ratio": _round_pair(
                bootstrap_mean_ci(ratio, reps=bootstrap_reps, rng_seed=rng_seed + 37)
            ),
        },
    }


def summarize_across_runs(run_summaries: list[dict[str, object]]) -> dict[str, object]:
    student = [float(run["means"]["student"]) for run in run_summaries]
    baseline = [float(run["means"]["baseline"]) for run in run_summaries]
    delta = [float(run["means"]["student_minus_baseline"]) for run in run_summaries]
    teacher = [float(run["means"]["teacher"]) for run in run_summaries]
    ratio = [float(run["means"]["student_teacher_ratio"]) for run in run_summaries]
    return {
        "runs": len(run_summaries),
        "mean": {
            "baseline": round(_mean(baseline), 6),
            "teacher": round(_mean(teacher), 6),
            "student": round(_mean(student), 6),
            "student_minus_baseline": round(_mean(delta), 6),
            "student_teacher_ratio": round(_mean(ratio), 6),
        },
        "std": {
            "baseline": round(_std(baseline), 6),
            "teacher": round(_std(teacher), 6),
            "student": round(_std(student), 6),
            "student_minus_baseline": round(_std(delta), 6),
            "student_teacher_ratio": round(_std(ratio), 6),
        },
    }


def summarize_pooled_rows(
    *,
    eval_rows_by_run: list[list[dict[str, Any]]],
    bootstrap_reps: int,
    rng_seed: int,
) -> dict[str, object]:
    pooled: list[dict[str, Any]] = []
    for rows in eval_rows_by_run:
        pooled.extend(rows)
    return summarize_eval_rows(eval_rows=pooled, bootstrap_reps=bootstrap_reps, rng_seed=rng_seed)


def should_early_stop_after_two_runs(
    *,
    first: dict[str, object],
    second: dict[str, object],
    threshold: float,
) -> dict[str, object]:
    first_student = float(first["means"]["student"])
    second_student = float(second["means"]["student"])
    first_delta = float(first["means"]["student_minus_baseline"])
    second_delta = float(second["means"]["student_minus_baseline"])
    delta_student = abs(first_student - second_student)
    delta_student_vs_baseline = abs(first_delta - second_delta)

    first_ci = first["ci95"]["student_minus_baseline"]
    second_ci = second["ci95"]["student_minus_baseline"]
    first_low, first_high = float(first_ci[0]), float(first_ci[1])
    second_low, second_high = float(second_ci[0]), float(second_ci[1])
    opposite_sign = (first_high < 0 and second_low > 0) or (second_high < 0 and first_low > 0)

    cond1 = delta_student <= threshold
    cond2 = delta_student_vs_baseline <= threshold
    cond3 = not opposite_sign
    return {
        "stop": cond1 and cond2 and cond3,
        "checks": {
            "student_gap_abs": round(delta_student, 6),
            "student_gap_threshold": threshold,
            "student_gap_ok": cond1,
            "student_minus_baseline_gap_abs": round(delta_student_vs_baseline, 6),
            "student_minus_baseline_gap_threshold": threshold,
            "student_minus_baseline_gap_ok": cond2,
            "directional_ci_stable": cond3,
            "first_student_minus_baseline_ci95": [round(first_low, 6), round(first_high, 6)],
            "second_student_minus_baseline_ci95": [round(second_low, 6), round(second_high, 6)],
        },
    }


def format_campaign_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Campaign Report",
        "",
        "## Budget",
        f"- Hard cap (USD): {summary['budget']['hard_cap_usd']:.2f}",
        f"- Prior spend (USD): {summary['budget']['prior_spend_usd']:.4f}",
        f"- New spend (USD): {summary['budget']['new_spend_usd']:.4f}",
        f"- Total spend incl prior (USD): {summary['budget']['total_spend_usd']:.4f}",
        f"- Stopped for budget: {summary['budget']['stopped_for_budget']}",
    ]
    if summary.get("stop_reason"):
        lines.append(f"- Stop reason: {summary['stop_reason']}")

    lines.extend(
        [
            "",
            "## Prompt Freeze",
            f"- Frozen prompt path: {summary['frozen_prompts']['frozen_path']}",
            f"- Source prompt path: {summary['frozen_prompts']['source_path']}",
            f"- SHA256: {summary['frozen_prompts']['sha256']}",
            f"- Total rows in source: {summary['frozen_prompts']['total_rows']}",
            f"- Prompt limit used: {summary['frozen_prompts']['prompt_limit']}",
            "",
            "## Per-Run Summary",
        ]
    )
    for run in summary["runs"]:
        lines.extend(
            [
                f"### Seed {run['seed']}",
                f"- Run dir: {run['run_dir']}",
                f"- Stages completed: {', '.join(run['stages_completed'])}",
                f"- Actual spend (USD): {run['actual_spend_usd']:.4f}",
            ]
        )
        metrics = run.get("metrics")
        if isinstance(metrics, dict):
            lines.extend(
                [
                    f"- Baseline mean: {metrics['means']['baseline']}",
                    f"- Teacher mean: {metrics['means']['teacher']}",
                    f"- Student mean: {metrics['means']['student']}",
                    f"- Student - Baseline mean: {metrics['means']['student_minus_baseline']}",
                    (
                        "- Student - Baseline 95% CI: "
                        f"{metrics['ci95']['student_minus_baseline'][0]} .. "
                        f"{metrics['ci95']['student_minus_baseline'][1]}"
                    ),
                ]
            )
        else:
            lines.append("- Metrics: not available (run stopped before eval).")
        lines.extend(
            [
                f"- Eval report: {run['artifacts']['eval_report']}",
                f"- Run audit report: {run['artifacts']['run_audit_report']}",
                f"- Eval rows: {run['artifacts']['eval_rows']}",
                f"- Ledger: {run['artifacts']['ledger']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            f"- Runs completed: {summary['aggregate']['across_runs']['runs']}",
            (
                "- Student mean across runs: "
                f"{summary['aggregate']['across_runs'].get('mean', {}).get('student', 'n/a')}"
            ),
            (
                "- Student std across runs: "
                f"{summary['aggregate']['across_runs'].get('std', {}).get('student', 'n/a')}"
            ),
            (
                "- Student-Baseline mean across runs: "
                f"{summary['aggregate']['across_runs'].get('mean', {}).get('student_minus_baseline', 'n/a')}"
            ),
            "- Pooled Student-Baseline 95% CI: "
            f"{summary['aggregate']['pooled'].get('ci95', {}).get('student_minus_baseline', ['n/a', 'n/a'])[0]} .. "
            f"{summary['aggregate']['pooled'].get('ci95', {}).get('student_minus_baseline', ['n/a', 'n/a'])[1]}",
            "",
            "## Early Stop Decision",
            f"- Triggered: {summary['early_stop']['triggered']}",
            f"- Evaluated after run count: {summary['early_stop']['evaluated_after_runs']}",
            f"- Student gap abs: {summary['early_stop']['checks'].get('student_gap_abs', 'n/a')}",
            (
                "- Student-Baseline gap abs: "
                f"{summary['early_stop']['checks'].get('student_minus_baseline_gap_abs', 'n/a')}"
            ),
            f"- Directional CI stable: {summary['early_stop']['checks'].get('directional_ci_stable', 'n/a')}",
            "",
        ]
    )
    return "\n".join(lines)


def bootstrap_mean_ci(values: list[float], *, reps: int, rng_seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(rng_seed)
    means: list[float] = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    return (_percentile(means, 2.5), _percentile(means, 97.5))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Prompt file row is not an object: {path}")
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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(fmean(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (sorted_values[lower] * (1 - weight)) + (sorted_values[upper] * weight)


def _round_pair(pair: tuple[float, float]) -> list[float]:
    return [round(pair[0], 6), round(pair[1], 6)]
