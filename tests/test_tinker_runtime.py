from inference_projects.config import load_config
from inference_projects.tinker_runtime import CANARY_FIXTURE_PATH, _cost_per_1k, load_canary_prompts


def test_canary_fixture_has_expected_prompt_count():
    rows = load_canary_prompts(fixture_path=CANARY_FIXTURE_PATH)
    assert len(rows) == 8
    assert all(row.prompt for row in rows)


def test_canary_fixture_limit_is_respected():
    rows = load_canary_prompts(limit=3, fixture_path=CANARY_FIXTURE_PATH)
    assert len(rows) == 3


def test_cost_per_1k_returns_zero_for_no_tokens():
    cfg = load_config()
    assert _cost_per_1k(cfg=cfg, prefill_tokens=0, sample_tokens=0) == 0.0


def test_cost_per_1k_positive_for_token_usage():
    cfg = load_config()
    value = _cost_per_1k(cfg=cfg, prefill_tokens=1000, sample_tokens=1000)
    assert value > 0
