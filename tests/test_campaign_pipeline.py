import json
from pathlib import Path
from typing import Callable

import pytest

from inference_projects.adapters import StageAdapters
from inference_projects.pipeline import run_pipeline_command
from inference_projects.tinker_runtime import REAL_USAGE_KEY


def _write_campaign_cap_config(tmp_path: Path, *, strict_run_cap: int = 1) -> Path:
    cfg_path = tmp_path / f"campaign_cap_{strict_run_cap}.toml"
    cfg_text = Path("config/default.toml").read_text().replace(
        "strict_run_cap = 16",
        f"strict_run_cap = {strict_run_cap}",
        1,
    )
    cfg_path.write_text(cfg_text)
    return cfg_path


def _usage(stage: str, seed: int, cost_usd: float) -> dict[str, object]:
    return {
        "prefill_tokens": 100,
        "sample_tokens": 30,
        "train_tokens": 0,
        "cost_usd": cost_usd,
        "provider_raw": {"stage": stage, "seed": seed},
        "run_id": f"{stage}-seed-{seed}",
    }


def _load_prompt_rows(cfg) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in Path(cfg.evaluation.prompt_file).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "reference": str(row.get("reference", "")),
            }
        )
    return rows


class FakeRealRL:
    mode = "real"

    def run(self, *, cfg, actual_cost_usd):
        _ = actual_cost_usd
        return {
            "model": cfg.teacher_model,
            "stage": "rl",
            "quality_score": 0.70,
            "stability_score": 0.91,
            "checkpoint_path": f"tinker://ckpt/rl/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/rl/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="rl", seed=cfg.seed, cost_usd=0.0101),
        }


class FakeRealTeacherFT:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "model": cfg.teacher_model,
            "stage": "teacher_ft",
            "quality_score": 0.72,
            "stability_score": 0.90,
            "checkpoint_path": f"tinker://ckpt/teacher_ft/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/teacher_ft/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="teacher_ft", seed=cfg.seed, cost_usd=0.0102),
        }


class FakeRealDistill:
    mode = "real"

    def run(self, *, cfg, teacher_payload, actual_cost_usd):
        _ = (teacher_payload, actual_cost_usd)
        return {
            "teacher_model": cfg.teacher_model,
            "student_model": cfg.student_model,
            "teacher_quality": 0.72,
            "student_quality": 0.70,
            "compression_ratio": 8.0,
            "stability_score": 0.88,
            "checkpoint_path": f"tinker://ckpt/distill/{cfg.seed}",
            "sampler_checkpoint_path": f"tinker://sampler/distill/{cfg.seed}",
            REAL_USAGE_KEY: _usage(stage="distill", seed=cfg.seed, cost_usd=0.0103),
        }


class FakeRealEval:
    mode = "real"

    def __init__(
        self,
        student_score_for_seed: Callable[[int], float],
        integrity_failed_for_seed: Callable[[int], bool] | None = None,
        integrity_status_for_seed: Callable[[int], str] | None = None,
    ):
        self._student_score_for_seed = student_score_for_seed
        self._integrity_failed_for_seed = integrity_failed_for_seed or (lambda seed: False)
        self._integrity_status_for_seed = integrity_status_for_seed

    def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
        _ = (teacher_payload, student_payload, actual_cost_usd)
        prompt_rows = _load_prompt_rows(cfg)
        baseline_score = 0.30
        teacher_score = 0.40
        student_score = float(self._student_score_for_seed(cfg.seed))
        eval_rows = []
        for row in prompt_rows:
            eval_rows.append(
                {
                    "row_id": row["id"],
                    "prompt": row["prompt"],
                    "reference": row["reference"],
                    "baseline_output": "baseline",
                    "teacher_output": "teacher",
                    "student_output": "student",
                    "baseline_overlap": baseline_score,
                    "teacher_overlap": teacher_score,
                    "student_overlap": student_score,
                    "student_vs_baseline_win": 1.0 if student_score > baseline_score else 0.0,
                    "student_vs_teacher_win": 1.0 if student_score > teacher_score else 0.0,
                }
            )

        student_retention = student_score / teacher_score if teacher_score else 0.0
        savings_pct = 30.0
        status = (
            self._integrity_status_for_seed(cfg.seed)
            if self._integrity_status_for_seed is not None
            else ("fail" if self._integrity_failed_for_seed(cfg.seed) else "pass")
        )
        return {
            "quality": {
                "benchmark": {
                    "baseline": baseline_score,
                    "teacher": teacher_score,
                    "student": student_score,
                    "student_retention_vs_teacher": round(student_retention, 4),
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": 1.0 if student_score > baseline_score else 0.0,
                    "student_vs_teacher_win_rate": 1.0 if student_score > teacher_score else 0.0,
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": 0.0010,
                    "student": 0.0007,
                    "student_savings_pct": savings_pct,
                },
                "eval_stage_cost_usd": 0.0104,
            },
            "training_stability": {
                "rl": {"stability_score": 0.91, "nan_events": 0},
                "teacher_ft": {"stability_score": 0.89, "nan_events": 0},
                "distill": {"stability_score": 0.88, "nan_events": 0},
            },
            "integrity": {
                "passed": status == "pass",
                "status": status,
                "reason": "forced-failure" if status == "fail" else ("forced-warning" if status == "warn" else ""),
                "checks": {"teacher_refusal_rate": 0.9 if self._integrity_failed_for_seed(cfg.seed) else 0.0},
            },
            "_eval_rows": eval_rows,
            REAL_USAGE_KEY: _usage(stage="eval", seed=cfg.seed, cost_usd=0.0104),
        }


def _patch_fake_real_adapters(
    monkeypatch: pytest.MonkeyPatch,
    student_score_for_seed: Callable[[int], float],
    integrity_failed_for_seed: Callable[[int], bool] | None = None,
    integrity_status_for_seed: Callable[[int], str] | None = None,
) -> None:
    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(
                student_score_for_seed=student_score_for_seed,
                integrity_failed_for_seed=integrity_failed_for_seed,
                integrity_status_for_seed=integrity_status_for_seed,
            ),
        ),
    )


def test_campaign_stops_after_two_runs_when_variance_low(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35 if seed == 17 else 0.36)

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["campaign_win"] is True
    assert "reproducibility" in summary
    assert summary["early_stop"]["triggered"] is True
    assert summary["executed_seeds"] == [17, 29]
    assert len(summary["runs"]) == 2
    assert all(run["metrics"]["rows"] == 150 for run in summary["runs"])
    summary_file = Path(summary["artifacts"]["campaign_summary"])
    report_file = Path(summary["artifacts"]["campaign_report"])
    assert summary_file.exists()
    assert report_file.exists()
    assert "artifacts" in json.loads(summary_file.read_text())

    first_rows = [json.loads(line) for line in Path(summary["runs"][0]["artifacts"]["eval_rows"]).read_text().splitlines() if line]
    second_rows = [json.loads(line) for line in Path(summary["runs"][1]["artifacts"]["eval_rows"]).read_text().splitlines() if line]
    assert [row["row_id"] for row in first_rows] == [row["row_id"] for row in second_rows]
    assert len(first_rows) == 150


def test_campaign_runs_third_seed_when_variance_high(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: {17: 0.35, 29: 0.50, 43: 0.37}[seed],
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["early_stop"]["triggered"] is False
    assert summary["executed_seeds"] == [17, 29, 43]
    assert len(summary["runs"]) == 3


def test_campaign_spend_fields_are_informational(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["spend"]["new_spend_usd"] >= 0.0


def test_campaign_integrity_failure_sets_needs_debug_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: 0.35,
        integrity_failed_for_seed=lambda seed: seed == 17,
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )

    assert summary is not None
    assert summary["campaign_status"] == "needs_debug"
    assert summary["executed_seeds"] == [17]
    assert summary["runs"][0]["integrity"]["passed"] is False
    assert summary["acceptance_checks"]["integrity_passed_all_runs"] is False


def test_campaign_integrity_warn_sets_needs_debug_without_forced_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(
        monkeypatch,
        student_score_for_seed=lambda seed: 0.35,
        integrity_status_for_seed=lambda seed: "warn",
    )

    summary = run_pipeline_command(
        "campaign",
        mode="real",
        state_dir=state_dir,
    )
    assert summary is not None
    assert summary["campaign_status"] == "needs_debug"
    assert all(run["integrity"]["status"] == "warn" for run in summary["runs"])


def test_campaign_enforces_strict_run_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_campaign_cap_config(tmp_path, strict_run_cap=1)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    assert summary["executed_seeds"] == [17]
    guardrails = summary["guardrails"]["campaign_strict_run_cap"]
    assert guardrails["configured"] == 1
    assert guardrails["enforced"] is True
    assert guardrails["effective_max_runs"] == 1
    assert guardrails["cap_hit"] is True


def test_campaign_ignores_evaluation_prompt_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "prompt_limit.toml"
    cfg_text = Path("config/default.toml").read_text().replace("prompt_limit = 150", "prompt_limit = 10", 1)
    cfg_path.write_text(cfg_text)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    assert summary["frozen_prompts"]["prompt_limit"] == 150
    assert all(run["metrics"]["rows"] == 150 for run in summary["runs"])


def test_campaign_marks_needs_debug_when_teacher_margin_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class LowTeacherEval(FakeRealEval):
        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            payload = super().run(
                cfg=cfg,
                teacher_payload=teacher_payload,
                student_payload=student_payload,
                actual_cost_usd=actual_cost_usd,
            )
            quality = payload["quality"]["benchmark"]
            quality["teacher"] = 0.33
            for row in payload["_eval_rows"]:
                row["teacher_overlap"] = 0.33
            return payload

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=LowTeacherEval(student_score_for_seed=lambda seed: 0.35),
        ),
    )

    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir)
    assert summary is not None
    assert summary["campaign_win"] is False
    assert summary["campaign_status"] == "needs_debug"
    assert summary["acceptance_checks"]["teacher_vs_baseline_margin_min_0_05_all_runs"] is False


def test_campaign_mock_reproducibility_mismatch_blocks_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = _write_campaign_cap_config(tmp_path, strict_run_cap=1)
    state_dir = tmp_path / "state"
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command("campaign", mode="mock", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    summary_path = Path(summary["artifacts"]["campaign_summary"])
    payload = json.loads(summary_path.read_text())
    payload["reproducibility"]["artifact_fingerprint"] = "tampered-artifact-fingerprint"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="reproducibility"):
        run_pipeline_command("campaign", mode="mock", state_dir=state_dir, config_path=cfg_path)


def test_campaign_real_reproducibility_invariant_mismatch_blocks_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_path = _write_campaign_cap_config(tmp_path, strict_run_cap=1)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")
    _patch_fake_real_adapters(monkeypatch, student_score_for_seed=lambda seed: 0.35)

    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    summary_path = Path(summary["artifacts"]["campaign_summary"])
    payload = json.loads(summary_path.read_text())
    payload["reproducibility"]["invariant_fingerprint"] = "tampered-invariant-fingerprint"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="reproducibility"):
        run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)


def test_campaign_marks_needs_debug_when_eval_runtime_threshold_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_path = _write_campaign_cap_config(tmp_path, strict_run_cap=1)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    import inference_projects.pipeline as pipeline_mod

    base_monotonic = pipeline_mod.time.monotonic
    monotonic_offset = {"seconds": 0.0}

    def shifted_monotonic() -> float:
        return base_monotonic() + float(monotonic_offset["seconds"])

    class RuntimeMissEval(FakeRealEval):
        def run(self, *, cfg, teacher_payload, student_payload, actual_cost_usd):
            payload = super().run(
                cfg=cfg,
                teacher_payload=teacher_payload,
                student_payload=student_payload,
                actual_cost_usd=actual_cost_usd,
            )
            # Simulate eval taking >720s without adding real wall-clock latency to the test.
            monotonic_offset["seconds"] = 721.0
            return payload

    monkeypatch.setattr("inference_projects.pipeline.time.monotonic", shifted_monotonic)
    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=RuntimeMissEval(student_score_for_seed=lambda seed: 0.35),
        ),
    )

    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    assert summary["campaign_win"] is False
    assert summary["campaign_status"] == "needs_debug"
    assert summary["acceptance_checks"]["eval_duration_under_720_seconds_all_runs"] is False
    assert summary["acceptance_checks"]["teacher_vs_baseline_margin_min_0_05_all_runs"] is True
    assert summary["acceptance_checks"]["integrity_passed_all_runs"] is True


def test_campaign_emits_summary_on_failure_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    cfg_path = tmp_path / "retry_once.toml"
    cfg_text = Path("config/default.toml").read_text().replace("max_consecutive_failures = 5", "max_consecutive_failures = 1")
    cfg_path.write_text(cfg_text)
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class BrokenDistill(FakeRealDistill):
        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            raise RuntimeError("forced-distill-failure")

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=FakeRealRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=BrokenDistill(),
            eval=FakeRealEval(student_score_for_seed=lambda seed: 0.35),
        ),
    )
    with pytest.raises(RuntimeError):
        run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    failed_summary = json.loads((state_dir / "campaign" / "campaign_summary.json").read_text())
    assert failed_summary["campaign_status"] == "failed"
    assert failed_summary["failure_class"] in {"failed", "invariant_failed", "transient_exhausted", "stalled"}

    call_counts = {"rl": 0, "teacher_ft": 0}

    class CountRL(FakeRealRL):
        def run(self, *, cfg, actual_cost_usd):
            call_counts["rl"] += 1
            return super().run(cfg=cfg, actual_cost_usd=actual_cost_usd)

    class CountTeacherFT(FakeRealTeacherFT):
        def run(self, *, cfg, teacher_payload, actual_cost_usd):
            call_counts["teacher_ft"] += 1
            return super().run(cfg=cfg, teacher_payload=teacher_payload, actual_cost_usd=actual_cost_usd)

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=CountRL(),
            teacher_ft=CountTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(student_score_for_seed=lambda seed: 0.35),
        ),
    )
    summary = run_pipeline_command("campaign", mode="real", state_dir=state_dir, config_path=cfg_path)
    assert summary is not None
    assert call_counts["rl"] == 1
    assert call_counts["teacher_ft"] == 1


def test_campaign_stall_timeout_marks_failed_and_emits_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "timeout.toml"
    cfg_text = Path("config/default.toml").read_text().replace("max_consecutive_failures = 5", "max_consecutive_failures = 1")
    cfg_path.write_text(cfg_text)
    state_dir = tmp_path / "state"
    monkeypatch.setenv("TINKER_API_KEY", "dummy")
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.test")

    class SlowRL(FakeRealRL):
        def run(self, *, cfg, actual_cost_usd):
            import time as _time

            _time.sleep(0.2)
            return super().run(cfg=cfg, actual_cost_usd=actual_cost_usd)

    monkeypatch.setattr(
        "inference_projects.pipeline.select_stage_adapters",
        lambda mode: StageAdapters(
            rl=SlowRL(),
            teacher_ft=FakeRealTeacherFT(),
            distill=FakeRealDistill(),
            eval=FakeRealEval(student_score_for_seed=lambda seed: 0.35),
        ),
    )
    with pytest.raises(RuntimeError):
        run_pipeline_command(
            "campaign",
            mode="real",
            state_dir=state_dir,
            config_path=cfg_path,
            progress_timeout_seconds=0.05,
        )
    summary = json.loads((state_dir / "campaign" / "campaign_summary.json").read_text())
    assert summary["failure_class"] == "stalled"
