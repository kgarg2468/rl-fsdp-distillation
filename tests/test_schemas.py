import pytest

from inference_projects.schemas import (
    SchemaValidationError,
    validate_audit_eval_row,
    validate_run_manifest_payload,
    validate_stage_audit_payload,
    validate_eval_metrics,
    validate_ledger_payload,
    validate_student_checkpoint,
    validate_teacher_checkpoint,
)


def test_teacher_schema_rejects_missing_required_fields():
    with pytest.raises(SchemaValidationError):
        validate_teacher_checkpoint({"schema_version": "1.0", "mode": "mock"})


def test_student_schema_rejects_missing_required_fields():
    with pytest.raises(SchemaValidationError):
        validate_student_checkpoint({"schema_version": "1.0", "mode": "mock"})


def test_eval_schema_rejects_missing_nested_fields():
    with pytest.raises(SchemaValidationError):
        validate_eval_metrics({"schema_version": "1.0", "mode": "mock", "quality": {}, "cost": {}, "training_stability": {}})


def test_ledger_schema_rejects_invalid_record_types():
    with pytest.raises(SchemaValidationError):
        validate_ledger_payload(
            {
                "schema_version": "1.0",
                "total_spend_usd": 1.0,
                "stage_spend_usd": {"rl": 1.0},
                "token_totals": {"prefill": 1, "sample": 1, "train": 1},
                "records": ["bad-record"],
            }
        )


def test_run_manifest_schema_rejects_missing_required_fields():
    with pytest.raises(SchemaValidationError):
        validate_run_manifest_payload({"schema_version": "1.0"})


def test_stage_audit_schema_rejects_missing_usage_block():
    with pytest.raises(SchemaValidationError):
        validate_stage_audit_payload(
            {
                "schema_version": "1.0",
                "mode": "real",
                "stage": "rl",
                "status": "completed",
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:01+00:00",
                "duration_seconds": 1.0,
                "projected_cost_usd": 0.1,
                "actual_cost_usd": 0.1,
                "stage_cap_usd": 1.0,
                "cumulative_total_before_usd": 0.0,
                "cumulative_total_after_usd": 0.1,
                "projected_tokens": {"prefill": 1, "sample": 1, "train": 0},
                "actual_tokens": {"prefill": 1, "sample": 1, "train": 0},
            }
        )


def test_eval_row_schema_rejects_missing_score_fields():
    with pytest.raises(SchemaValidationError):
        validate_audit_eval_row(
            {
                "row_id": "x",
                "prompt": "p",
                "reference": "r",
                "baseline_output": "a",
                "teacher_output": "b",
                "student_output": "c",
            }
        )
