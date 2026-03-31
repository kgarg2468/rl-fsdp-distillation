from __future__ import annotations

from inference_projects.config import ProjectConfig
from inference_projects.pricing import TokenUsage, cost_usd


class BudgetExceededError(RuntimeError):
    """Raised when any budget boundary is exceeded."""


def stage_token_usage(stage: str, cfg: ProjectConfig) -> TokenUsage:
    if stage not in cfg.token_caps:
        raise KeyError(f"Unknown stage: {stage}")
    return cfg.token_caps[stage]


def projected_stage_cost_usd(stage: str, cfg: ProjectConfig) -> float:
    usage = stage_token_usage(stage, cfg)
    rates = cfg.token_rates_per_million
    return cost_usd(
        usage,
        prefill_rate=rates["prefill"],
        sample_rate=rates["sample"],
        train_rate=rates["train"],
    )


def projected_total_cost_usd(cfg: ProjectConfig) -> float:
    return round(sum(projected_stage_cost_usd(stage, cfg) for stage in cfg.token_caps), 4)


def ensure_within_stage_budget(stage: str, stage_cost_usd: float, cfg: ProjectConfig) -> None:
    stage_cap = cfg.budget.stage_budgets_usd[stage]
    if stage_cost_usd > stage_cap:
        raise BudgetExceededError(
            f"Stage '{stage}' projected ${stage_cost_usd:.2f}, exceeds stage cap ${stage_cap:.2f}."
        )


def ensure_within_hard_cap(*, current_total: float, incoming_cost: float, cfg: ProjectConfig) -> None:
    projected_total = current_total + incoming_cost
    if projected_total > cfg.budget.hard_cap_usd:
        raise BudgetExceededError(
            f"Projected total ${projected_total:.2f} exceeds hard cap ${cfg.budget.hard_cap_usd:.2f}."
        )
