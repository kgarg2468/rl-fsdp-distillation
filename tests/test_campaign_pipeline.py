import json
from pathlib import Path
from typing import Callable

import pytest

from inference_projects.adapters import StageAdapters
from inference_projects.pipeline import run_pipeline_command
from inference_projects.tinker_runtime import REAL_USAGE_KEY


def _usage(stage: str, seed: int, cost_usd: float) -> dict[str, object]:
    return {
        "prefill_tokens": 100,
        "sample_tokens": 30,
        "train_tokens": 0,
        "cost_usd": cost_usd,
        "provider_raw": {"stage": stage, "seed": seed},
        "run_id": f"{stage}-seed-{seed}",
    }


def _load_prompt_rows(cfg) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in Path(cfg.evaluation.prompt_file).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "reference": str(row.get("reference", "")),
            }
        )
        if len(rows) >= cfg.evaluation.prompt_limit:
            break
    return rows


class FakeRealRL:
    mode = "real"

    def run(self, *, cfg, actual_cost_usd):
        _ = actual_cost_usd
        return {
            "model": cfg.teacher_model,
            "stage": "rl",
            "quality_score": 0.70,
            "stability_score": 0.91,
            "checkpoint_path": f"tinker://ckpt/rl/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/rl/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="rl", seed=cfg.seed, cost_usd=0.0101),
        }


class FakeRealFSDP:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "model": cfg.teacher_model,
            "stage": "fsdp",
            "quality_score": 0.72,
            "stability_score": 0.90,
            "checkpoint_path": f"tinker://ckpt/fsdp/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/fsdp/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="fsdp", seed=cfg.seed, cost_usd=0.0102),
        }


class FakeRealDistill:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "teacher_model": cfg.teacher_model,
            "student_model": cfg.student_model,
            "teacher_quality": 0.72,
            "student_quality": 0.70,
            "compression_ratio": 8.0,
            "stability_score": 0.88,
            "checkpoint_path": f"tinker://ckpt/distill/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/distill/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="distill", seed=cfg.seed, cost_usd=0.0103),
        }


class FakeRealEval:
    mode = "real"

    def __init__(
        self,
        student_score_for_seed: Callable[[int], float],
        integrity_failed_for_seed: Callable[[int], bool] | None = None,
        integrity_status_for_seed: Callable[[int], str] | None = None,
    ):
        self._student_score_for_seed = student_score_for_seed
        self._integrity_failed_for_seed = integrity_failed_for_seed or (lambda seed: False)
        self._integrity_status_for_seed = integrity_status_for_seed

    def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
        _ = (teacher_payload, student_payload, actual_cost_usd)
        prompt_rows = _load_prompt_rows(cfg)
        baseline_score = 0.30
        teacher_score = 0.40
        student_score = float(self._student_score_for_seed(cfg.seed))
        eval_rows = []
        for row in prompt_rows:
            eval_rows.append(
                {
                    "row_id": row["id"],
                    "prompt": row["prompt"],
                    "reference": row["reference"],
                    "baseline_output": "baseline",
                    "teacher_output": "teacher",
                    "student_output": "student",
                    "baseline_overlap": baseline_score,
                    "teacher_overlap": teacher_score,
                    "student_overlap": student_score,
                    "student_vs_baseline_win": 1.0 if student_score > baseline_score else 0.0,
                    "student_vs_teacher_win": 1.0 if student_score > teacher_score else 0.0,
                }
            )

        student_retention = student_score / teacher_score if teacher_score else 0.0
        savings_pct = 30.0
        status = (
            self._integrity_status_for_seed(cfg.seed)
            if self._integrity_status_for_seed is not None
            else ("fail" if self._integrity_failed_for_seed(cfg.seed) else "pass")
        )
        return {
            "quality": {
                "benchmark": {
                    "baseline": baseline_score,
                    "teacher": teacher_score,
                    "student": student_score,
                    "student_retention_vs_teacher": round(student_retention, 4),
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": 1.0 if student_score > baseline_score else 0.0,
                    "student_vs_teacher_win_rate": 1.0 if student_score > teacher_score else 0.0,
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": 0.0010,
                    "student": 0.0007,
                    "student_savings_pct": savings_pct,
                },
                "eval_stage_cost_usd": 0.0104,
            },
            "training_stability": {
                "rl": {"stability_score": 0.91, "nan_events": 0},
                "fsdp": {"stability_score": 0.89, "nan_events": 0},
                "distill": {"stability_score": 0.88, "nan_events": 0},
            },
            "integrity": {
                "passed": status == "pass",
                "status": status,
                "reason": "forced-failure" if status == "fail" else ("forced-warning" if status == "warn" else ""),
                "checks": {"teacher_refusal_rate": 0.9 if self._integrity_failed_for_seed(cfg.seed) else 0.0},
            },
            "_eval_rows": eval_rows,
            REAL_USAGE_KEY: _usage(stage="eval", seed=cfg.seed, cost_usd=0.0104),
        }


def _patch_fake_real_adapters(
    monkeypatch: pytest.MonkeyPatch,
    student_score_for_seed: Callable[[int], float],
    integrity_failed_for_seed: Callable[[int], bool] | None = None,
    integrity_status_for_seed: Callable[[int], str] | None = None,
) -> None:
    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            fsdp=FakeRealFSDP(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(
                student_score_for_seed=student_score_for_seed,
                integrity_failed_for_seed=integrity_failed_for_seed,
                integrity_status_for_seed=integrity_status_for_seed,
            ),
        ),
    )


def test_campaign_stops_after_two_runs_when_variance_low(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35 if seed == 17 else 0.36)

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["early_stop"]["triggered"] is True
    assert summary["executed_seeds"] == [17, 29]
    assert len(summary["runs"]) == 2
    assert all(run["metrics"]["rows"] == 150 for run in summary["runs"])
    summary_file = Path(summary["artifacts"]["campaign_summary"])
    report_file = Path(summary["artifacts"]["campaign_report"])
    assert summary_file.exists()
    assert report_file.exists()
    assert "artifacts" in json.loads(summary_file.read_text())

    first_rows = [json.loads(line) for line in Path(summary["runs"][0]["artifacts"]["eval_rows"]).read_text().splitlines() if line]
    second_rows = [json.loads(line) for line in Path(summary["runs"][1]["artifacts"]["eval_rows"]).read_text().splitlines() if line]
    assert [row["row_id"] for row in first_rows] == [row["row_id"] for row in second_rows]
    assert len(first_rows) == 150


def test_campaign_runs_third_seed_when_variance_high(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: {17: 0.35, 29: 0.50, 43: 0.37}[seed],
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["early_stop"]["triggered"] is False
    assert summary["executed_seeds"] == [17, 29, 43]
    assert len(summary["runs"]) == 3


def test_campaign_spend_fields_are_informational(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["spend"]["new_spend_usd"] >= 0.0


def test_campaign_integrity_failure_sets_needs_debug_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: 0.35,
        integrity_failed_for_seed=lambda seed: seed == 17,
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["campaign_status"] == "needs_debug"
    assert summary["executed_seeds"] == [17]
    assert summary["runs"][0]["integrity"]["passed"] is False
    assert summary["acceptance_checks"]["integrity_passed_all_runs"] is False


def test_campaign_integrity_warn_sets_needs_debug_without_forced_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: 0.35,
        integrity_status_for_seed=lambda seed: "warn",
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["campaign_status"] == "needs_debug"
    assert all(run["integrity"]["status"] == "warn" for run in summary["runs"])
