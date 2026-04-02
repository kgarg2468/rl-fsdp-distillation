from __future__ import annotations

from inference_projects.config import ProjectConfig
from inference_projects.pricing import TokenUsage, cost_usd

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
