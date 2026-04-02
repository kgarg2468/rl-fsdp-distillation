import json
from pathlib import Path
from typing import Literal

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


class FakeRealTeacherFT:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "model": cfg.teacher_model,
            "stage": "teacher_ft",
            "quality_score": 0.72,
            "stability_score": 0.90,
            "checkpoint_path": f"tinker://ckpt/teacher_ft/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/teacher_ft/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="teacher_ft", seed=cfg.seed, cost_usd=0.0102),
        }


class FakeRealDistill:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "teacher_model": cfg.teacher_model,
            "student_model": cfg.student_model,
            "teacher_quality": 0.75,
            "student_quality": 0.71,
            "compression_ratio": 8.0,
            "stability_score": 0.88,
            "checkpoint_path": f"tinker://ckpt/distill/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/distill/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="distill", seed=cfg.seed, cost_usd=0.0103),
        }


class FakeRealEval:
    mode = "real"

    def __init__(self, *, regime: Literal["good", "flat"]):
        self._regime = regime

    def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
        _ = (teacher_payload, student_payload, actual_cost_usd)
        prompt_rows = _load_prompt_rows(cfg)

        baseline = 0.45
        teacher = 0.52
        if cfg.distillation.teacher_prompt_template == "numeric_strict":
            teacher += 0.01

        if self._regime == "flat":
            student = 0.45
        else:
            gain = 0.01
            if cfg.distillation.filter_profile == "strict":
                gain += 0.01
            if cfg.distillation.kd_alpha >= 0.7:
                gain += 0.01
            if cfg.distillation.lora_rank >= 16:
                gain += 0.01
            student = min(0.98, baseline + gain)

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
                    "baseline_overlap": baseline,
                    "teacher_overlap": teacher,
                    "student_overlap": student,
                    "baseline_exact_match": baseline,
                    "teacher_exact_match": teacher,
                    "student_exact_match": student,
                    "baseline_numeric_parse": 1.0,
                    "teacher_numeric_parse": 1.0,
                    "student_numeric_parse": 1.0,
                    "student_vs_baseline_win": 1.0 if student > baseline else 0.0,
                    "student_vs_teacher_win": 1.0 if student > teacher else 0.0,
                }
            )

        return {
            "quality": {
                "benchmark": {
                    "baseline": baseline,
                    "teacher": teacher,
                    "student": student,
                    "student_retention_vs_teacher": round(student / teacher, 4),
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": 1.0 if student > baseline else 0.0,
                    "student_vs_teacher_win_rate": 1.0 if student > teacher else 0.0,
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": 0.0010,
                    "student": 0.0007,
                    "student_savings_pct": 30.0,
                },
                "eval_stage_cost_usd": 0.0104,
            },
            "training_stability": {
                "rl": {"stability_score": 0.91, "nan_events": 0},
                "teacher_ft": {"stability_score": 0.89, "nan_events": 0},
                "distill": {"stability_score": 0.88, "nan_events": 0},
            },
            "integrity": {
                "passed": True,
                "status": "ok",
                "reason": "",
                "checks": {"teacher_refusal_rate": 0.0},
            },
            "_eval_rows": eval_rows,
            REAL_USAGE_KEY: _usage(stage="eval", seed=cfg.seed, cost_usd=0.0104),
        }


def _patch_fake_real_adapters(monkeypatch: pytest.MonkeyPatch, *, regime: Literal["good", "flat"]) -> None:
    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(regime=regime),
        ),
    )


def test_tune_real_mode_generates_summary_and_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, regime="good")

    summary = run_pipeline_command(
        "tune",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["status"] == "ok"
    assert summary["teacher_sweep"]["runs_executed"] == 8
    assert summary["distill_sweep"]["runs_executed"] == 8
    assert summary["final_campaign"]["executed"] is True
    assert "acceptance_checks" in summary
    assert summary["acceptance_checks"]["teacher_margin_winner_pass"] is True
    assert summary["acceptance_checks"]["student_gain_winner_pass"] is True

    tuning_summary = Path(summary["artifacts"]["tuning_summary"])
    tuning_report = Path(summary["artifacts"]["tuning_report"])
    candidates = Path(summary["artifacts"]["candidates"])
    assert tuning_summary.exists()
    assert tuning_report.exists()
    assert candidates.exists()

    first_candidate = json.loads(candidates.read_text().splitlines()[0])
    assert first_candidate["aggregation_runs"] == 6
    assert "teacher_margin_pass" in first_candidate
    assert "student_gain_pass" in first_candidate
    assert "integrity_pass" in first_candidate


def test_tune_marks_needs_debug_when_strict_gate_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, regime="flat")

    summary = run_pipeline_command(
        "tune",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["status"] == "needs_debug"
    assert summary["final_campaign"]["executed"] is False


def test_tune_reports_spend_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, regime="good")

    summary = run_pipeline_command(
        "tune",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["status"] == "ok"
    assert "spend" in summary
    assert summary["spend"]["total_spend_usd"] >= 0.0


def test_tune_strict_run_cap_enforced_with_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "tight_cap.toml"
    cfg_text = Path("config/default.toml").read_text()
    prefix, suffix = cfg_text.split("[tuning]", 1)
    suffix = suffix.replace("strict_run_cap = 16", "strict_run_cap = 1", 1)
    cfg_text = prefix + "[tuning]" + suffix
    cfg_path.write_text(cfg_text)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, regime="good")

    with pytest.raises(RuntimeError):
        run_pipeline_command(
            "tune",
            mode="real",
            state_dir=state_dir,
            config_path=cfg_path,
        )

    summary_path = state_dir / "tuning" / "tuning_summary.json"
    report_path = state_dir / "tuning" / "tuning_report.md"
    candidates_path = state_dir / "tuning" / "candidates.jsonl"
    assert summary_path.exists()
    assert report_path.exists()
    assert candidates_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["status"] == "failed"
    assert payload["failure_class"] in {"invariant_failed", "failed"}
