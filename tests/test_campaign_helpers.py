import json
from pathlib import Path

from inference_projects.campaign import (
    bootstrap_mean_ci,
    freeze_prompt_file,
    should_early_stop_after_two_runs,
    summarize_across_runs,
    summarize_eval_rows,
)


def test_bootstrap_mean_ci_is_deterministic():
    values = [0.1, 0.2, 0.5, 0.7, 0.9]
    first = bootstrap_mean_ci(values, reps=500, rng_seed=19)
    second = bootstrap_mean_ci(values, reps=500, rng_seed=19)
    assert first == second
    assert first[0] <= first[1]


def test_freeze_prompt_file_uses_all_rows_and_validates_uniqueness(tmp_path: Path):
    source = tmp_path / "prompts.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prompt": "p1", "reference": "r1"}),
                json.dumps({"id": "b", "prompt": "p2", "reference": "r2"}),
                json.dumps({"id": "c", "prompt": "p3", "reference": "r3"}),
            ]
        )
        + "\n"
    )
    frozen = tmp_path / "frozen.jsonl"
    info = freeze_prompt_file(source_path=source, frozen_path=frozen, prompt_limit=2)
    assert frozen.exists()
    assert info.total_rows == 3
    assert info.prompt_limit == 3
    assert info.prompt_ids == ["a", "b", "c"]


def test_summarize_across_runs_computes_std():
    run_a = {
        "means": {
            "baseline": 0.2,
            "teacher": 0.3,
            "student": 0.4,
            "student_minus_baseline": 0.2,
            "student_teacher_ratio": 1.2,
        }
    }
    run_b = {
        "means": {
            "baseline": 0.25,
            "teacher": 0.35,
            "student": 0.45,
            "student_minus_baseline": 0.2,
            "student_teacher_ratio": 1.1,
        }
    }
    summary = summarize_across_runs([run_a, run_b])
    assert summary["runs"] == 2
    assert summary["mean"]["student"] == 0.425
    assert summary["std"]["student"] > 0


def test_early_stop_checks_threshold_and_direction():
    run_one = {
        "means": {"student": 0.60, "student_minus_baseline": 0.08},
        "ci95": {"student_minus_baseline": [0.01, 0.15]},
    }
    run_two = {
        "means": {"student": 0.62, "student_minus_baseline": 0.07},
        "ci95": {"student_minus_baseline": [0.00, 0.14]},
    }
    decision = should_early_stop_after_two_runs(first=run_one, second=run_two, threshold=0.03)
    assert decision["stop"] is True


def test_summarize_eval_rows_has_expected_shape():
    rows = [
        {"baseline_overlap": 0.2, "teacher_overlap": 0.4, "student_overlap": 0.3},
        {"baseline_overlap": 0.1, "teacher_overlap": 0.2, "student_overlap": 0.3},
    ]
    summary = summarize_eval_rows(eval_rows=rows, bootstrap_reps=300, rng_seed=11)
    assert summary["rows"] == 2
    assert "means" in summary
    assert "ci95" in summary
