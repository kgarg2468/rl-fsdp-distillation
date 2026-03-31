from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"


class SchemaValidationError(ValueError):
    """Raised when artifact payload does not match required schema."""


def validate_teacher_checkpoint(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {
            "schema_version": str,
            "mode": str,
            "model": str,
            "stage": str,
            "quality_score": (int, float),
            "stability_score": (int, float),
        },
    )


def validate_student_checkpoint(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {
            "schema_version": str,
            "mode": str,
            "teacher_model": str,
            "student_model": str,
            "teacher_quality": (int, float),
            "student_quality": (int, float),
            "compression_ratio": (int, float),
            "stability_score": (int, float),
        },
    )


def validate_eval_metrics(payload: dict[str, Any]) -> None:
    _require_keys(payload, {"schema_version": str, "mode": str, "quality": dict, "cost": dict, "training_stability": dict})

    benchmark = _nested(payload, "quality", "benchmark")
    _require_keys(
        benchmark,
        {
            "baseline": (int, float),
            "teacher": (int, float),
            "student": (int, float),
            "student_retention_vs_teacher": (int, float),
        },
    )

    llm_judge = _nested(payload, "quality", "llm_judge")
    _require_keys(
        llm_judge,
        {
            "student_vs_baseline_win_rate": (int, float),
            "student_vs_teacher_win_rate": (int, float),
        },
    )

    inference = _nested(payload, "cost", "inference_usd_per_1k_tokens")
    _require_keys(
        inference,
        {
            "teacher": (int, float),
            "student": (int, float),
            "student_savings_pct": (int, float),
        },
    )


def validate_ledger_payload(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {
            "schema_version": str,
            "total_spend_usd": (int, float),
            "stage_spend_usd": dict,
            "token_totals": dict,
            "records": list,
        },
    )
    token_totals = payload["token_totals"]
    _require_keys(token_totals, {"prefill": int, "sample": int, "train": int})

    for record in payload["records"]:
        if not isinstance(record, dict):
            raise SchemaValidationError("Each ledger record must be a dict")
        _require_keys(
            record,
            {
                "mode": str,
                "stage": str,
                "projected_cost_usd": (int, float),
                "actual_cost_usd": (int, float),
                "projected_tokens": dict,
                "actual_tokens": dict,
                "status": str,
            },
        )


def _nested(payload: dict[str, Any], key: str, nested_key: str) -> dict[str, Any]:
    outer = payload.get(key)
    if not isinstance(outer, dict):
        raise SchemaValidationError(f"Expected '{key}' to be an object")
    inner = outer.get(nested_key)
    if not isinstance(inner, dict):
        raise SchemaValidationError(f"Expected '{key}.{nested_key}' to be an object")
    return inner


def _require_keys(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, expected_type in expected.items():
        if key not in payload:
            raise SchemaValidationError(f"Missing required key: {key}")
        if not isinstance(payload[key], expected_type):
            raise SchemaValidationError(
                f"Invalid type for '{key}': expected {expected_type}, got {type(payload[key])}"
            )
