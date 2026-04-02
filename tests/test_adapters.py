import pytest

from inference_projects import adapters
from inference_projects.adapters import RealDistillAdapter, RealEvalAdapter, RealTeacherFTAdapter, RealRLAdapter, select_stage_adapters
from inference_projects.config import load_config
from inference_projects.preflight import run_preflight
from inference_projects.tinker_runtime import REAL_USAGE_KEY


def test_select_stage_adapters_for_mock_mode():
    adapters = select_stage_adapters("mock")
    assert adapters.rl.mode == "mock"
    assert adapters.teacher_ft.mode == "mock"
    assert adapters.distill.mode == "mock"
    assert adapters.eval.mode == "mock"


def test_select_stage_adapters_for_real_mode():
    adapters = select_stage_adapters("real")
    assert adapters.rl.mode == "real"
    assert adapters.teacher_ft.mode == "real"
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


def _usage_stub() -> dict[str, object]:
    return {
        "prefill_tokens": 10,
        "sample_tokens": 5,
        "train_tokens": 0,
        "cost_usd": None,
        "provider_raw": {"source": "test"},
        "run_id": "run-test",
    }


def test_real_rl_adapter_returns_schema_payload_with_usage(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    monkeypatch.setattr(
        adapters,
        "run_real_rl",
        lambda *, cfg: {
            "payload": {
                "model": cfg.teacher_model,
                "stage": "rl",
                "quality_score": 0.71,
                "stability_score": 0.93,
            },
            "usage": _usage_stub(),
        },
    )
    payload = RealRLAdapter().run(cfg=cfg, actual_cost_usd=0.0)
    assert payload["model"] == cfg.teacher_model
    assert payload["stage"] == "rl"
    assert isinstance(payload["quality_score"], float)
    assert isinstance(payload["stability_score"], float)
    assert REAL_USAGE_KEY in payload


def test_real_teacher_ft_adapter_returns_schema_payload_with_usage(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    monkeypatch.setattr(
        adapters,
        "run_real_teacher_ft",
        lambda *, cfg, teacher_payload: {
            "payload": {
                "model": cfg.teacher_model,
                "stage": "teacher_ft",
                "quality_score": float(teacher_payload["quality_score"]) + 0.01,
                "stability_score": 0.90,
            },
            "usage": _usage_stub(),
        },
    )
    payload = RealTeacherFTAdapter().run(
        cfg=cfg,
        teacher_payload={"quality_score": 0.71},
        actual_cost_usd=0.0,
    )
    assert payload["stage"] == "teacher_ft"
    assert isinstance(payload["quality_score"], float)
    assert isinstance(payload["stability_score"], float)
    assert REAL_USAGE_KEY in payload


def test_real_distill_adapter_returns_schema_payload_with_usage(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    monkeypatch.setattr(
        adapters,
        "run_real_distill",
        lambda *, cfg, teacher_payload: {
            "payload": {
                "teacher_model": cfg.teacher_model,
                "student_model": cfg.student_model,
                "teacher_quality": float(teacher_payload["quality_score"]),
                "student_quality": 0.69,
                "compression_ratio": 8.0,
                "stability_score": 0.89,
            },
            "usage": _usage_stub(),
        },
    )
    payload = RealDistillAdapter().run(
        cfg=cfg,
        teacher_payload={"quality_score": 0.72},
        actual_cost_usd=0.0,
    )
    assert payload["teacher_model"] == cfg.teacher_model
    assert payload["student_model"] == cfg.student_model
    assert isinstance(payload["student_quality"], float)
    assert REAL_USAGE_KEY in payload


def test_real_eval_adapter_returns_schema_payload_with_usage(monkeypatch: pytest.MonkeyPatch):
    cfg = load_config()
    monkeypatch.setattr(
        adapters,
        "run_real_eval",
        lambda *, cfg, teacher_payload, student_payload: {
            "payload": {
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
            },
            "usage": _usage_stub(),
        },
    )
    payload = RealEvalAdapter().run(
        cfg=cfg,
        teacher_payload={"quality_score": 0.74},
        student_payload={"student_quality": 0.70},
        actual_cost_usd=0.0,
    )
    assert "quality" in payload
    assert "cost" in payload
    assert "training_stability" in payload
    assert REAL_USAGE_KEY in payload
