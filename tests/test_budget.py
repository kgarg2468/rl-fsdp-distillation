from pathlib import Path

import pytest

from inference_projects import budget
from inference_projects.config import load_config
from inference_projects.pricing import TokenUsage, cost_usd


def test_cost_usd_from_token_usage():
    usage = TokenUsage(prefill=1_000_000, sample=2_000_000, train=3_000_000)
    value = cost_usd(usage, prefill_rate=0.13, sample_rate=0.40, train_rate=0.40)
    assert value == pytest.approx(2.13)


def test_stage_projection_matches_default_config():
    cfg = load_config(Path("config/default.toml"))
    projected = budget.projected_stage_cost_usd("rl", cfg)
    assert projected == pytest.approx(8.39, abs=0.01)


def test_token_usage_for_known_stage():
    cfg = load_config(Path("config/default.toml"))
    usage = budget.stage_token_usage("distill", cfg)
    assert usage.prefill == 2_000_000
    assert usage.sample == 8_000_000
    assert usage.train == 4_000_000


def test_unknown_stage_rejected():
    cfg = load_config(Path("config/default.toml"))
    with pytest.raises(KeyError):
        budget.stage_token_usage("unknown", cfg)


def test_default_total_projection_within_target_window():
    cfg = load_config(Path("config/default.toml"))
    total = budget.projected_total_cost_usd(cfg)
    assert 20.0 <= total <= 30.0
