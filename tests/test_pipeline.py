import json
from pathlib import Path

import pytest
from inference_projects.adapters import StageAdapters
from inference_projects.preflight import SetupError
from inference_projects.pipeline import run_pipeline_command
from inference_projects.tinker_runtime import REAL_USAGE_KEY


def test_rl_stage_creates_teacher_checkpoint(tmp_path: Path):
    state_dir = tmp_path / "state"
    run_pipeline_command("rl", state_dir=state_dir)
    ckpt = state_dir / "artifacts/checkpoints/teacher/best_checkpoint.json"
    ledger = state_dir / "artifacts/ledger.json"
    assert ckpt.exists()
    assert ledger.exists()


def test_all_command_creates_report_and_eval(tmp_path: Path):
    state_dir = tmp_path / "state"
    run_pipeline_command("all", state_dir=state_dir, mode="mock")
    assert (state_dir / "artifacts/eval/eval_metrics.json").exists()
    report = state_dir / "artifacts/reports/eval_report.md"
    assert report.exists()
    text = report.read_text()
    assert "## Run Metadata" in text
    assert "Run mode: mock" in text
    assert "## Disclaimer" in text
    assert "## Quality" in text
    assert "## Cost" in text
    assert "## Training Stability" in text
    assert "Projected spend (USD)" in text
    assert "Actual spend (USD)" in text


def test_projection_warning_band_does_not_block_run(tmp_path: Path):
    state_dir = tmp_path / "state"
    cfg_path = tmp_path / "warn_only.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("projection_warning_min_usd = 20.0", "projection_warning_min_usd = 5.0")
    text = text.replace("projection_warning_max_usd = 30.0", "projection_warning_max_usd = 10.0")
    cfg_path.write_text(text)
    run_pipeline_command("rl", config_path=cfg_path, state_dir=state_dir)
    assert (state_dir / "artifacts/checkpoints/teacher/best_checkpoint.json").exists()


def test_stage_budget_cap_exceeded_blocks_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    cfg_path = tmp_path / "tight_caps.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("rl_prefill = 3000000", "rl_prefill = 10")
    text = text.replace("rl_sample = 12000000", "rl_sample = 10")
    cfg_path.write_text(text)
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class FakeRealRL:
        mode = "real"

        def run(self, *, cfg, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            return {
                "model": "meta-llama/Llama-3.1-8B",
                "stage": "rl",
                "quality_score": 0.75,
                "stability_score": 0.92,
                REAL_USAGE_KEY: {
                    "prefill_tokens": 321,
                    "sample_tokens": 123,
                    "train_tokens": 0,
                    "cost_usd": 0.0456,
                    "provider_raw": {"mocked": True},
                    "run_id": "run-abc",
                },
            }

    class NoopTeacherFT:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopDistill:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopEval:
        mode = "real"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=NoopTeacherFT(),
            distill=NoopDistill(),
            eval=NoopEval(),
        ),
    )
    with pytest.raises(RuntimeError) as exc:
        run_pipeline_command("rl", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert "budget cap" in str(exc.value).lower()
    assert not (state_dir / "artifacts/checkpoints/teacher/best_checkpoint.json").exists()


def test_real_mode_without_credentials_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    for key in ("TINKER_API_KEY", "TINKER_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SetupError) as exc:
        run_pipeline_command("rl", mode="real", state_dir=state_dir)
    assert "Missing required environment variable for real mode" in str(exc.value)


def test_dryrun_does_not_create_artifacts(tmp_path: Path):
    state_dir = tmp_path / "state"
    result = run_pipeline_command("dryrun", mode="mock", state_dir=state_dir)
    assert result is not None
    assert "projected_total_usd" in result
    assert not (state_dir / "artifacts").exists()


def test_real_mode_uses_adapter_usage_for_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class FakeRealRL:
        mode = "real"

        def run(self, *, cfg, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            return {
                "model": "meta-llama/Llama-3.1-8B",
                "stage": "rl",
                "quality_score": 0.75,
                "stability_score": 0.92,
                REAL_USAGE_KEY: {
                    "prefill_tokens": 321,
                    "sample_tokens": 123,
                    "train_tokens": 0,
                    "cost_usd": 0.0456,
                    "provider_raw": {"mocked": True},
                    "run_id": "run-abc",
                },
            }

    class NoopTeacherFT:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopDistill:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopEval:
        mode = "real"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=NoopTeacherFT(),
            distill=NoopDistill(),
            eval=NoopEval(),
        ),
    )
    run_pipeline_command("rl", mode="real", state_dir=state_dir)

    ledger = (state_dir / "artifacts/ledger.json").read_text()
    assert "\"actual_cost_usd\": 0.0456" in ledger
    assert "\"prefill\": 321" in ledger
    assert "\"sample\": 123" in ledger


def test_real_mode_missing_usage_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class BadRealRL:
        mode = "real"

        def run(self, *, cfg, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            return {
                "model": "meta-llama/Llama-3.1-8B",
                "stage": "rl",
                "quality_score": 0.75,
                "stability_score": 0.92,
            }

    class NoopTeacherFT:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopDistill:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopEval:
        mode = "real"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=BadRealRL(),
            teacher_ft=NoopTeacherFT(),
            distill=NoopDistill(),
            eval=NoopEval(),
        ),
    )
    with pytest.raises(RuntimeError):
        run_pipeline_command("rl", mode="real", state_dir=state_dir)


def test_campaign_runs_without_prior_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "inference_projects.pipeline.run_campaign",
        lambda *, cfg, mode, state_dir, **kwargs: {
            "executed_seeds": [17],
            "mode": mode,
            "state_dir": str(state_dir),
        },
    )
    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir)
    assert summary is not None
    assert "executed_seeds" in summary


def test_tune_runs_without_prior_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setattr(
        "inference_projects.pipeline.run_tune",
        lambda *, cfg, mode, state_dir, **kwargs: {
            "status": "ok",
            "mode": mode,
            "state_dir": str(state_dir),
        },
    )
    summary = run_pipeline_command("tune", mode="real", state_dir=state_dir)
    assert summary is not None
    assert "status" in summary


def test_all_timeout_writes_failure_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "timeout.toml"
    cfg_text = Path("config/default.toml").read_text().replace("max_consecutive_failures = 5", "max_consecutive_failures = 1")
    cfg_path.write_text(cfg_text)
    state_dir = tmp_path / "state"

    class SlowRL:
        mode = "mock"

        def run(self, *, cfg, actual_cost_usd):
            import time as _time

            _ = (cfg, actual_cost_usd)
            _time.sleep(0.2)
            return {
                "model": "meta-llama/Llama-3.1-8B",
                "stage": "rl",
                "quality_score": 0.75,
                "stability_score": 0.9,
            }

    class NoopTeacherFT:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopDistill:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    class NoopEval:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            raise AssertionError("unexpected call")

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=SlowRL(),
            teacher_ft=NoopTeacherFT(),
            distill=NoopDistill(),
            eval=NoopEval(),
        ),
    )
    with pytest.raises(RuntimeError):
        run_pipeline_command(
            "all",
            mode="mock",
            state_dir=state_dir,
            config_path=cfg_path,
            progress_timeout_seconds=0.05,
        )
    failures = state_dir / "artifacts" / "audit" / "all_failures.jsonl"
    assert failures.exists()
    latest = json.loads((state_dir / "artifacts" / "audit" / "all_failure_latest.json").read_text())
    assert latest["stage"] == "rl"
    assert latest["failure_class"] == "stalled"


def test_all_resume_skips_completed_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    calls = {"rl": 0, "teacher_ft": 0, "distill": 0, "eval": 0}

    class CountRL:
        mode = "mock"

        def run(self, *, cfg, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            calls["rl"] += 1
            return {
                "model": "meta-llama/Llama-3.1-8B",
                "stage": "rl",
                "quality_score": 0.72,
                "stability_score": 0.91,
            }

    class CountTeacherFT:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            calls["teacher_ft"] += 1
            updated = dict(teacher_payload)
            updated.update(
                {
                    "stage": "teacher_ft",
                    "quality_score": 0.74,
                    "stability_score": 0.89,
                    "axolotl_teacher_ft": True,
                }
            )
            return updated

    class CountDistill:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            _ = (cfg, teacher_payload, actual_cost_usd)
            calls["distill"] += 1
            return {
                "teacher_model": "meta-llama/Llama-3.1-8B",
                "student_model": "meta-llama/Llama-3.2-1B",
                "teacher_quality": 0.74,
                "student_quality": 0.70,
                "compression_ratio": 8.0,
                "stability_score": 0.90,
            }

    class CountEval:
        mode = "mock"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            _ = (cfg, teacher_payload, student_payload, actual_cost_usd)
            calls["eval"] += 1
            return {
                "quality": {
                    "benchmark": {
                        "baseline": 0.61,
                        "teacher": 0.74,
                        "student": 0.70,
                        "student_retention_vs_teacher": 0.9459,
                    },
                    "llm_judge": {
                        "student_vs_baseline_win_rate": 0.66,
                        "student_vs_teacher_win_rate": 0.44,
                    },
                },
                "cost": {
                    "inference_usd_per_1k_tokens": {
                        "teacher": 0.00027,
                        "student": 0.00011,
                        "student_savings_pct": 59.26,
                    },
                    "eval_stage_cost_usd": 0.0,
                },
                "training_stability": {
                    "rl": {"stability_score": 0.91, "nan_events": 0},
                    "teacher_ft": {"stability_score": 0.89, "nan_events": 0},
                    "distill": {"stability_score": 0.90, "nan_events": 0},
                },
            }

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=CountRL(),
            teacher_ft=CountTeacherFT(),
            distill=CountDistill(),
            eval=CountEval(),
        ),
    )
    run_pipeline_command("all", mode="mock", state_dir=state_dir, resume=True)
    run_pipeline_command("all", mode="mock", state_dir=state_dir, resume=True)
    assert calls == {"rl": 1, "teacher_ft": 1, "distill": 1, "eval": 1}


def test_all_real_integrity_warn_fails_with_integrity_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    def _usage(run_id: str, *, train_tokens: int = 10) -> dict[str, object]:
        return {
            "prefill_tokens": 10,
            "sample_tokens": 10,
            "train_tokens": train_tokens,
            "cost_usd": 0.001,
            "provider_raw": {"mocked": True},
            "run_id": run_id,
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
                REAL_USAGE_KEY: _usage("run-rl"),
            }

    class FakeRealTeacherFT:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            _ = (cfg, actual_cost_usd)
            updated = dict(teacher_payload)
            updated.update(
                {
                    "stage": "teacher_ft",
                    "quality_score": 0.76,
                    "stability_score": 0.91,
                    "axolotl_teacher_ft": True,
                    REAL_USAGE_KEY: _usage("run-teacher"),
                }
            )
            return updated

    class FakeRealDistill:
        mode = "real"

        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            _ = (cfg, teacher_payload, actual_cost_usd)
            return {
                "teacher_model": "meta-llama/Llama-3.1-8B",
                "student_model": "meta-llama/Llama-3.2-1B",
                "teacher_quality": 0.76,
                "student_quality": 0.70,
                "compression_ratio": 8.0,
                "stability_score": 0.90,
                REAL_USAGE_KEY: _usage("run-distill"),
            }

    class FakeRealEvalWarn:
        mode = "real"

        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            _ = (cfg, teacher_payload, student_payload, actual_cost_usd)
            return {
                "quality": {
                    "benchmark": {
                        "baseline": 0.8,
                        "teacher": 0.6,
                        "student": 0.7,
                        "student_retention_vs_teacher": 1.1667,
                    },
                    "llm_judge": {
                        "student_vs_baseline_win_rate": 0.4,
                        "student_vs_teacher_win_rate": 0.6,
                    },
                },
                "cost": {
                    "inference_usd_per_1k_tokens": {
                        "teacher": 0.0003,
                        "student": 0.0002,
                        "student_savings_pct": 33.33,
                    },
                    "eval_stage_cost_usd": 0.001,
                },
                "training_stability": {
                    "rl": {"stability_score": 0.92, "nan_events": 0},
                    "teacher_ft": {"stability_score": 0.91, "nan_events": 0},
                    "distill": {"stability_score": 0.90, "nan_events": 0},
                },
                "integrity": {
                    "passed": False,
                    "status": "warn",
                    "reason": "teacher overlap below baseline",
                    "checks": {"teacher_overlap_score": 0.6},
                },
                REAL_USAGE_KEY: _usage("run-eval", train_tokens=0),
            }

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEvalWarn(),
        ),
    )
    with pytest.raises(RuntimeError):
        run_pipeline_command("all", mode="real", state_dir=state_dir)
    latest = json.loads((state_dir / "artifacts" / "audit" / "all_failure_latest.json").read_text())
    assert latest["stage"] == "eval"
    assert latest["failure_class"] == "integrity_failed"
