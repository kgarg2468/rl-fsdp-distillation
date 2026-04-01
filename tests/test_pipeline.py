from pathlib import Path

import pytest
from inference_projects.preflight import SetupError
from inference_projects.pipeline import run_pipeline_command


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


def test_low_hard_cap_aborts_run(tmp_path: Path):
    state_dir = tmp_path / "state"
    cfg_path = tmp_path / "tiny_cap.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("target_cap_usd = 25.0", "target_cap_usd = 0.5")
    text = text.replace("hard_cap_usd = 30.0", "hard_cap_usd = 1.0")
    cfg_path.write_text(text)
    with pytest.raises(SetupError):
        run_pipeline_command("rl", config_path=cfg_path, state_dir=state_dir)


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
