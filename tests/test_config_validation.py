from pathlib import Path

import pytest

from inference_projects.config import load_config


def test_invalid_real_polling_values_rejected(tmp_path: Path):
    cfg = tmp_path / "bad_runtime.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("real_poll_interval_seconds = 15", "real_poll_interval_seconds = 0")
    cfg.write_text(text)
    with pytest.raises(ValueError):
        load_config(cfg)


def test_real_canary_config_loads_successfully():
    cfg = load_config("config/real_canary.toml")
    assert cfg.runtime.real_required_env == ("TINKER_API_KEY", "TINKER_BASE_URL")
    assert cfg.runtime.real_poll_interval_seconds == 15
    assert cfg.runtime.real_poll_timeout_seconds == 3600
    assert cfg.evaluation.prompt_limit == 150
    assert cfg.evaluation.max_concurrency > 0
    assert cfg.evaluation.batch_size > 0
    assert cfg.evaluation.max_tokens_eval > 0
    assert cfg.evaluation.eval_temperature == 0.0
    assert cfg.evaluation.eval_max_tokens_candidates == (48, 96)
    assert cfg.distillation.filter_profile == "moderate"
    assert cfg.distillation.lora_rank == 8
    assert cfg.campaign.seeds == (17, 29, 43)
    assert cfg.tuning.sweep_runs == 16
    assert cfg.tuning.teacher_candidates == (cfg.teacher_model,)


def test_campaign_max_runs_cannot_exceed_seed_count(tmp_path: Path):
    cfg_path = tmp_path / "bad_campaign.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("max_runs = 3", "max_runs = 4")
    cfg_path.write_text(text)
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_empty_teacher_candidates_rejected(tmp_path: Path):
    cfg_path = tmp_path / "bad_tuning.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("teacher_candidates = [\"meta-llama/Llama-3.1-8B\"]", "teacher_candidates = []")
    cfg_path.write_text(text)
    with pytest.raises(ValueError):
        load_config(cfg_path)
