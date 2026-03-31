import os

import pytest

from inference_projects.adapters import select_stage_adapters
from inference_projects.config import load_config
from inference_projects.preflight import run_preflight


def test_select_stage_adapters_for_mock_mode():
    adapters = select_stage_adapters("mock")
    assert adapters.rl.mode == "mock"
    assert adapters.fsdp.mode == "mock"
    assert adapters.distill.mode == "mock"
    assert adapters.eval.mode == "mock"


def test_select_stage_adapters_for_real_mode():
    adapters = select_stage_adapters("real")
    assert adapters.rl.mode == "real"
    assert adapters.fsdp.mode == "real"
    assert adapters.distill.mode == "real"
    assert adapters.eval.mode == "real"


def test_real_preflight_reports_missing_env_vars(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    for key in cfg.runtime.real_required_env:
        monkeypatch.delenv(key, raising=False)
    result = run_preflight(mode="real", cfg=cfg)
    assert not result.ok
    assert any("Missing required environment variable" in msg for msg in result.errors)


def test_real_preflight_passes_with_required_env(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    for key in cfg.runtime.real_required_env:
        monkeypatch.setenv(key, f"dummy-{key.lower()}")
    result = run_preflight(mode="real", cfg=cfg)
    assert result.ok


def test_mock_preflight_does_not_require_real_env(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    for key in cfg.runtime.real_required_env:
        monkeypatch.delenv(key, raising=False)
    result = run_preflight(mode="mock", cfg=cfg)
    assert result.ok
