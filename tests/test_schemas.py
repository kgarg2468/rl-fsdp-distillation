import pytest

from inference_projects.schemas import (
    SchemaValidationError,
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
