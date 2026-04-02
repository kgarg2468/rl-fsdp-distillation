from pathlib import Path

from inference_projects.tuning import (
    candidate_acceptance,
    distill_l8_candidates,
    freeze_prompt_slice,
    promote_candidates,
    rank_teacher_candidates,
    teacher_headroom_candidates,
)


def test_teacher_headroom_candidates_are_deterministic():
    rows = teacher_headroom_candidates(
        current_teacher="meta-llama/Llama-3.1-8B",
        stronger_teacher="meta-llama/Llama-3.1-70B-Instruct",
        max_tokens_candidates=(48, 96),
    )
    assert len(rows) == 8
    assert rows[0]["candidate_id"] == "teacher-01"
    assert {row["teacher_model"] for row in rows} == {
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B-Instruct",
    }
    assert {row["eval_temperature"] for row in rows} == {0.0, 0.2}
    assert {row["max_tokens_eval"] for row in rows} == {48, 96}


def test_distill_l8_candidates_cover_expected_knobs():
    rows = distill_l8_candidates()
    assert len(rows) == 8
    assert {row["filter_profile"] for row in rows} == {"moderate", "strict"}
    assert {row["hard_example_ratio"] for row in rows} == {0.4, 0.7}
    assert {row["kd_alpha"] for row in rows} == {0.3, 0.7}
    assert {row["kd_temperature"] for row in rows} == {1.0, 4.0}
    assert {row["learning_rate"] for row in rows} == {0.00001, 0.00005}
    assert {row["epochs"] for row in rows} == {1, 3}
    assert {row["lora_rank"] for row in rows} == {8, 16}


def test_candidate_acceptance_applies_strict_thresholds():
    metrics = {
        "means": {
            "baseline": 0.50,
            "teacher": 0.56,
            "student": 0.54,
        }
    }
    checks = candidate_acceptance(metrics=metrics, integrity_passed=True)
    assert checks == {
        "teacher_margin_pass": True,
        "student_gain_pass": True,
        "integrity_pass": True,
    }


def test_rank_and_promote_candidates():
    teacher_ranked = rank_teacher_candidates(
        [
            {"candidate_id": "a", "teacher_minus_baseline": 0.01, "teacher_score": 0.60, "integrity_pass": True},
            {"candidate_id": "b", "teacher_minus_baseline": 0.07, "teacher_score": 0.65, "integrity_pass": True},
            {"candidate_id": "c", "teacher_minus_baseline": 0.20, "teacher_score": 0.30, "integrity_pass": False},
        ]
    )
    assert [row["candidate_id"] for row in teacher_ranked] == ["b", "a"]

    promoted = promote_candidates(
        [
            {
                "candidate_id": "x",
                "teacher_margin_pass": True,
                "student_gain_pass": True,
                "integrity_pass": True,
                "student_minus_baseline": 0.04,
                "student_minus_baseline_exact_match": 0.03,
                "eval_duration_seconds": 500.0,
            },
            {
                "candidate_id": "y",
                "teacher_margin_pass": True,
                "student_gain_pass": True,
                "integrity_pass": True,
                "student_minus_baseline": 0.05,
                "student_minus_baseline_exact_match": 0.02,
                "eval_duration_seconds": 400.0,
            },
            {
                "candidate_id": "z",
                "teacher_margin_pass": False,
                "student_gain_pass": True,
                "integrity_pass": True,
                "student_minus_baseline": 0.09,
                "student_minus_baseline_exact_match": 0.06,
                "eval_duration_seconds": 300.0,
            },
        ],
        top_k=2,
    )
    assert [row["candidate_id"] for row in promoted] == ["y", "x"]


def test_freeze_prompt_slice_writes_limited_rows(tmp_path: Path):
    source = Path("src/inference_projects/fixtures/real_eval_prompts_150.jsonl")
    out = tmp_path / "frozen_30.jsonl"
    frozen = freeze_prompt_slice(source_path=source, frozen_path=out, prompt_limit=30)
    assert out.exists()
    assert frozen.rows == 30
    assert len(frozen.prompt_ids) == 30
    assert frozen.prompt_ids[0] == "p001"
    assert frozen.prompt_ids[-1] == "p030"
