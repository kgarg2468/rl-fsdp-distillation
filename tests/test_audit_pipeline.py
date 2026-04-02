import json
from pathlib import Path

import pytest

from inference_projects.adapters import StageAdapters
from inference_projects.pipeline import run_pipeline_command
from inference_projects.tinker_runtime import REAL_USAGE_KEY


def _usage(run_id: str, stage: str, cost_usd: float) -> dict[str, object]:
    return {
        "prefill_tokens": 111,
        "sample_tokens": 22,
        "train_tokens": 0,
        "cost_usd": cost_usd,
        "provider_raw": {"request_id": f"req-{stage}", "stage": stage},
        "run_id": run_id,
    }


def _fake_trace(stage: str, model_label: str, row_id: str) -> dict[str, object]:
    return {
        "row_id": row_id,
        "prompt": f"prompt-{row_id}",
        "reference": f"reference-{row_id}",
        "output": f"output-{row_id}",
        "prefill_tokens": 4,
        "sample_tokens": 2,
        "stage": stage,
        "model_label": model_label,
    }


def _fake_eval_row(row_id: str) -> dict[str, object]:
    return {
        "row_id": row_id,
        "prompt": f"prompt-{row_id}",
        "reference": f"reference-{row_id}",
        "baseline_output": "baseline-out",
        "teacher_output": "teacher-out",
        "student_output": "student-out",
        "baseline_overlap": 0.1,
        "teacher_overlap": 0.3,
        "student_overlap": 0.4,
        "student_vs_baseline_win": 1.0,
        "student_vs_teacher_win": 1.0,
    }


class FakeRealRL:
    mode = "real"

    def run(self, *, cfg, actual_cost_usd):
        _ = (cfg, actual_cost_usd)
        return {
            "model": "meta-llama/Llama-3.1-8B",
            "stage": "rl",
            "quality_score": 0.75,
            "stability_score": 0.92,
            "checkpoint_path": "tinker://checkpoints/rl-1",
            "run_id": "run-rl-1",
            "_prompt_traces": [_fake_trace(stage="rl", model_label="teacher", row_id="rl-1")],
            REAL_USAGE_KEY: _usage(run_id="run-rl-1", stage="rl", cost_usd=0.1001),
        }


class FakeRealTeacherFT:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (cfg, actual_cost_usd, teacher_payload)
        return {
            "model": "meta-llama/Llama-3.1-8B",
            "stage": "teacher_ft",
            "quality_score": 0.76,
            "stability_score": 0.90,
            "checkpoint_path": "tinker://checkpoints/teacher_ft-1",
            "run_id": "run-teacher_ft-1",
            "_prompt_traces": [_fake_trace(stage="teacher_ft", model_label="teacher", row_id="teacher_ft-1")],
            REAL_USAGE_KEY: _usage(run_id="run-teacher_ft-1", stage="teacher_ft", cost_usd=0.2002),
        }


class FakeRealDistill:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (cfg, actual_cost_usd, teacher_payload)
        return {
            "teacher_model": "meta-llama/Llama-3.1-8B",
            "student_model": "meta-llama/Llama-3.2-1B",
            "teacher_quality": 0.76,
            "student_quality": 0.71,
            "compression_ratio": 8.0,
            "stability_score": 0.88,
            "checkpoint_path": "tinker://checkpoints/distill-1",
            "run_id": "run-distill-1",
            "_prompt_traces": [
                _fake_trace(stage="distill", model_label="teacher", row_id="distill-t-1"),
                _fake_trace(stage="distill", model_label="student", row_id="distill-s-1"),
            ],
            REAL_USAGE_KEY: _usage(run_id="run-distill-1", stage="distill", cost_usd=0.3003),
        }


class FakeRealEval:
    mode = "real"

    def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
        _ = (cfg, teacher_payload, student_payload, actual_cost_usd)
        return {
            "quality": {
                "benchmark": {
                    "baseline": 0.62,
                    "teacher": 0.74,
                    "student": 0.70,
                    "student_retention_vs_teacher": 0.9459,
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": 0.65,
                    "student_vs_teacher_win_rate": 0.45,
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": 0.001,
                    "student": 0.0007,
                    "student_savings_pct": 30.0,
                },
                "eval_stage_cost_usd": 0.05,
            },
            "training_stability": {
                "rl": {"stability_score": 0.91, "nan_events": 0},
                "teacher_ft": {"stability_score": 0.89, "nan_events": 0},
                "distill": {"stability_score": 0.88, "nan_events": 0},
            },
            "_prompt_traces": [
                _fake_trace(stage="eval", model_label="baseline", row_id="eval-1"),
                _fake_trace(stage="eval", model_label="teacher", row_id="eval-1"),
                _fake_trace(stage="eval", model_label="student", row_id="eval-1"),
            ],
            "_eval_rows": [_fake_eval_row("eval-1"), _fake_eval_row("eval-2")],
            REAL_USAGE_KEY: _usage(run_id="run-eval-1", stage="eval", cost_usd=0.4004),
        }


def _patch_fake_real_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(),
        ),
    )


def test_real_all_writes_audit_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch)

    run_pipeline_command("all", mode="real", state_dir=state_dir)

    assert (state_dir / "artifacts/audit/run_manifest.json").exists()
    assert (state_dir / "artifacts/audit/stage_rl.json").exists()
    assert (state_dir / "artifacts/audit/stage_teacher_ft.json").exists()
    assert (state_dir / "artifacts/audit/stage_distill.json").exists()
    assert (state_dir / "artifacts/audit/stage_eval.json").exists()
    assert (state_dir / "artifacts/audit/eval_rows.jsonl").exists()
    assert (state_dir / "artifacts/reports/run_audit_report.md").exists()


def test_stage_audit_persists_provider_usage_and_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch)

    run_pipeline_command("all", mode="real", state_dir=state_dir)

    stage_rl = json.loads((state_dir / "artifacts/audit/stage_rl.json").read_text())
    assert stage_rl["stage"] == "rl"
    assert stage_rl["usage"]["run_id"] == "run-rl-1"
    assert stage_rl["usage"]["provider_raw"]["request_id"] == "req-rl"
    assert stage_rl["status"] == "completed"
    assert stage_rl["started_at"]
    assert stage_rl["finished_at"]
    assert stage_rl["duration_seconds"] >= 0


def test_eval_rows_contains_full_trace_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch)

    run_pipeline_command("all", mode="real", state_dir=state_dir)

    lines = (state_dir / "artifacts/audit/eval_rows.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert [row["row_id"] for row in rows] == ["eval-1", "eval-2"]
    assert all("baseline_output" in row for row in rows)
    assert all("teacher_output" in row for row in rows)
    assert all("student_output" in row for row in rows)
    assert all("baseline_overlap" in row for row in rows)
    assert all("teacher_overlap" in row for row in rows)
    assert all("student_overlap" in row for row in rows)


def test_run_audit_report_has_required_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch)

    run_pipeline_command("all", mode="real", state_dir=state_dir)

    text = (state_dir / "artifacts/reports/run_audit_report.md").read_text()
    assert "## Run Metadata" in text
    assert "## Stage Chronology" in text
    assert "## Prompt-Level Evidence Appendix" in text
    assert "## Anomalies and Notes" in text
