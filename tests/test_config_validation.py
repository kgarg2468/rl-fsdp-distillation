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
    assert cfg.runtime.retry_max_connections == 1000
    assert cfg.runtime.retry_enabled is True
    assert cfg.runtime.max_consecutive_failures == 5
    assert cfg.evaluation.prompt_limit == 150
    assert cfg.evaluation.max_concurrency > 0
    assert cfg.evaluation.batch_size > 0
    assert cfg.evaluation.max_tokens_eval > 0
    assert cfg.evaluation.eval_temperature == 0.0
    assert cfg.evaluation.eval_max_tokens_candidates == (16, 32)
    assert cfg.evaluation.teacher_integrity_numeric_parse_threshold == 0.8
    assert cfg.distillation.training_prompt_limit == 150
    assert cfg.distillation.training_prompt_file is None
    assert cfg.distillation.filter_profile == "strict"
    assert cfg.distillation.lora_rank == 8
    assert cfg.campaign.seeds == (17, 29, 43)
    assert cfg.campaign.strict_run_cap == 16
    assert cfg.tuning.sweep_runs == 16
    assert cfg.tuning.strict_run_cap == 16
    assert cfg.tuning.teacher_candidates == (cfg.teacher_model,)


def test_phase1_canary_config_loads_successfully():
    cfg = load_config("config/real_canary_phase1.toml")
    assert cfg.campaign.seeds == (42,)
    assert cfg.campaign.min_runs == 1
    assert cfg.campaign.max_runs == 1
    assert cfg.campaign.strict_run_cap == 1
    assert cfg.distillation.filter_profile == "strict"


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


def test_invalid_retry_config_rejected(tmp_path: Path):
    cfg_path = tmp_path / "bad_retry.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("retry_delay_max_seconds = 10.0", "retry_delay_max_seconds = 0.1")
    cfg_path.write_text(text)
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_distillation_training_prompt_file_resolves_relative_path(tmp_path: Path):
    cfg_path = tmp_path / "with_training_prompt.toml"
    prompt_path = tmp_path / "curated_train.jsonl"
    prompt_path.write_text('{"id":"r1","prompt":"1+1","reference":"2"}\n')
    text = Path("config/default.toml").read_text()
    text = text.replace("training_prompt_limit = 150", 'training_prompt_limit = 150\ntraining_prompt_file = "curated_train.jsonl"')
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)
    assert cfg.distillation.training_prompt_file == prompt_path.resolve()


def test_distillation_training_prompt_file_must_exist(tmp_path: Path):
    cfg_path = tmp_path / "missing_training_prompt.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace(
        "training_prompt_limit = 150",
        'training_prompt_limit = 150\ntraining_prompt_file = "does_not_exist.jsonl"',
    )
    cfg_path.write_text(text)
    with pytest.raises(FileNotFoundError):
        load_config(cfg_path)


def test_cap_like_fields_allow_zero_and_warning_band_order_is_not_enforced(tmp_path: Path):
    cfg_path = tmp_path / "uncapped.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("projection_warning_min_usd = 20.0", "projection_warning_min_usd = 40.0")
    text = text.replace("projection_warning_max_usd = 30.0", "projection_warning_max_usd = 10.0")
    text = text.replace("prompt_limit = 150", "prompt_limit = 0", 1)
    text = text.replace("training_prompt_limit = 150", "training_prompt_limit = 0", 1)
    text = text.replace("strict_run_cap = 16", "strict_run_cap = 0", 1)
    text = text.replace("stage1_prompt_limit = 30", "stage1_prompt_limit = 0", 1)
    text = text.replace("stage2_prompt_limit = 150", "stage2_prompt_limit = 0", 1)
    text = text.replace("strict_run_cap = 16", "strict_run_cap = 0", 1)
    cfg_path.write_text(text)

    cfg = load_config(cfg_path)
    assert cfg.evaluation.prompt_limit == 0
    assert cfg.distillation.training_prompt_limit == 0
    assert cfg.campaign.strict_run_cap == 0
    assert cfg.tuning.stage1_prompt_limit == 0
    assert cfg.tuning.stage2_prompt_limit == 0
    assert cfg.tuning.strict_run_cap == 0


def test_negative_cap_like_fields_are_rejected(tmp_path: Path):
    cfg_path = tmp_path / "negative.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("prompt_limit = 150", "prompt_limit = -1", 1)
    cfg_path.write_text(text)
    with pytest.raises(ValueError):
        load_config(cfg_path)
