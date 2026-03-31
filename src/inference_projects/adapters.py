from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from inference_projects.config import ProjectConfig


class RLStageAdapter(Protocol):
    mode: str

    def run(self, *, cfg: ProjectConfig, actual_cost_usd: float) -> dict[str, object]:
        ...


class FSDPStageAdapter(Protocol):
    mode: str

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        ...


class DistillStageAdapter(Protocol):
    mode: str

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        ...


class EvalStageAdapter(Protocol):
    mode: str

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        student_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class StageAdapters:
    rl: RLStageAdapter
    fsdp: FSDPStageAdapter
    distill: DistillStageAdapter
    eval: EvalStageAdapter


@dataclass(frozen=True)
class MockRLAdapter:
    mode: str = "mock"

    def run(self, *, cfg: ProjectConfig, actual_cost_usd: float) -> dict[str, object]:
        return {
            "model": cfg.teacher_model,
            "stage": "rl",
            "quality_score": 0.72,
            "stability_score": 0.91,
            "cost_usd": actual_cost_usd,
            "notes": "Mock Tinker RL training completed under budget guardrails.",
        }


@dataclass(frozen=True)
class MockFSDPAdapter:
    mode: str = "mock"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        updated = dict(teacher_payload)
        updated.update(
            {
                "stage": "fsdp",
                "quality_score": 0.74,
                "stability_score": 0.89,
                "axolotl_fsdp": True,
                "fsdp_cost_usd": actual_cost_usd,
                "notes": "Mock Axolotl FSDP fine-tuning completed.",
            }
        )
        return updated


@dataclass(frozen=True)
class MockDistillAdapter:
    mode: str = "mock"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        return {
            "teacher_model": cfg.teacher_model,
            "student_model": cfg.student_model,
            "teacher_quality": float(teacher_payload.get("quality_score", 0.74)),
            "student_quality": 0.70,
            "compression_ratio": 8.0,
            "stability_score": 0.90,
            "distill_cost_usd": actual_cost_usd,
            "notes": "Mock teacher->student distillation completed.",
        }


@dataclass(frozen=True)
class MockEvalAdapter:
    mode: str = "mock"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        student_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        teacher_quality = float(teacher_payload.get("quality_score", 0.74))
        student_quality = float(student_payload.get("student_quality", 0.70))
        baseline_quality = 0.61

        teacher_cost_per_1k = 0.00027
        student_cost_per_1k = 0.00011

        return {
            "quality": {
                "benchmark": {
                    "baseline": baseline_quality,
                    "teacher": teacher_quality,
                    "student": student_quality,
                    "student_retention_vs_teacher": round(student_quality / teacher_quality, 4),
                },
                "llm_judge": {
                    "student_vs_baseline_win_rate": 0.66,
                    "student_vs_teacher_win_rate": 0.44,
                },
            },
            "cost": {
                "inference_usd_per_1k_tokens": {
                    "teacher": teacher_cost_per_1k,
                    "student": student_cost_per_1k,
                    "student_savings_pct": round((1 - (student_cost_per_1k / teacher_cost_per_1k)) * 100, 2),
                },
                "eval_stage_cost_usd": actual_cost_usd,
            },
            "training_stability": {
                "rl": {"stability_score": 0.91, "nan_events": 0},
                "fsdp": {"stability_score": 0.89, "nan_events": 0},
                "distill": {"stability_score": 0.90, "nan_events": 0},
            },
        }


@dataclass(frozen=True)
class RealRLAdapter:
    mode: str = "real"

    def run(self, *, cfg: ProjectConfig, actual_cost_usd: float) -> dict[str, object]:
        raise NotImplementedError(
            "Real RL adapter scaffolded. Configure Tinker integration and replace this method implementation."
        )


@dataclass(frozen=True)
class RealFSDPAdapter:
    mode: str = "real"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        raise NotImplementedError(
            "Real FSDP adapter scaffolded. Configure Axolotl launch integration and replace this implementation."
        )


@dataclass(frozen=True)
class RealDistillAdapter:
    mode: str = "real"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        raise NotImplementedError(
            "Real distillation adapter scaffolded. Configure teacher->student data generation and training integration."
        )


@dataclass(frozen=True)
class RealEvalAdapter:
    mode: str = "real"

    def run(
        self,
        *,
        cfg: ProjectConfig,
        teacher_payload: dict[str, object],
        student_payload: dict[str, object],
        actual_cost_usd: float,
    ) -> dict[str, object]:
        raise NotImplementedError(
            "Real eval adapter scaffolded. Configure benchmark + LLM-judge integrations and replace this implementation."
        )


def select_stage_adapters(mode: str) -> StageAdapters:
    if mode == "mock":
        return StageAdapters(
            rl=MockRLAdapter(),
            fsdp=MockFSDPAdapter(),
            distill=MockDistillAdapter(),
            eval=MockEvalAdapter(),
        )
    if mode == "real":
        return StageAdapters(
            rl=RealRLAdapter(),
            fsdp=RealFSDPAdapter(),
            distill=RealDistillAdapter(),
            eval=RealEvalAdapter(),
        )
    raise ValueError(f"Unsupported mode: {mode}")
