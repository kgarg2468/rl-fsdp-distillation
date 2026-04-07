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
