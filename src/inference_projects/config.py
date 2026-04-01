from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from inference_projects.pricing import TokenUsage

REQUIRED_STAGES = ("rl", "fsdp", "distill", "eval")


@dataclass(frozen=True)
class BudgetConfig:
    target_cap_usd: float
    hard_cap_usd: float
    stage_budgets_usd: dict[str, float]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    teacher_model: str
    student_model: str
    baseline_model: str
    token_rates_per_million: dict[str, float]
    token_caps: dict[str, TokenUsage]
    budget: BudgetConfig
    runtime: "RuntimeConfig"
    evaluation: "EvaluationConfig"
    campaign: "CampaignConfig"


@dataclass(frozen=True)
class RuntimeConfig:
    default_mode: str
    projection_warning_min_usd: float
    projection_warning_max_usd: float
    real_required_env: tuple[str, ...]
    real_poll_interval_seconds: int
    real_poll_timeout_seconds: int


@dataclass(frozen=True)
class EvaluationConfig:
    prompt_file: Path
    prompt_limit: int


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    min_runs: int
    max_runs: int
    bootstrap_reps: int
    early_stop_threshold: float


def _token_caps_by_stage(raw: dict[str, int]) -> dict[str, TokenUsage]:
    out: dict[str, TokenUsage] = {}
    for stage in REQUIRED_STAGES:
        required_keys = (f"{stage}_prefill", f"{stage}_sample", f"{stage}_train")
        for key in required_keys:
            if key not in raw:
                raise ValueError(f"Missing token cap key: {key}")
        out[stage] = TokenUsage(
            prefill=int(raw[f"{stage}_prefill"]),
            sample=int(raw[f"{stage}_sample"]),
            train=int(raw[f"{stage}_train"]),
        )
    return out


def _validate_budget(budget: BudgetConfig) -> None:
    if budget.target_cap_usd > budget.hard_cap_usd:
        raise ValueError("target_cap_usd cannot be greater than hard_cap_usd")
    for stage in REQUIRED_STAGES:
        if stage not in budget.stage_budgets_usd:
            raise ValueError(f"Missing stage budget for '{stage}'")
        if budget.stage_budgets_usd[stage] <= 0:
            raise ValueError(f"Stage budget for '{stage}' must be > 0")
    if sum(budget.stage_budgets_usd.values()) < budget.target_cap_usd:
        raise ValueError("Stage budgets should sum to at least target_cap_usd")


def _validate_runtime(runtime: RuntimeConfig) -> None:
    if runtime.default_mode not in {"mock", "real"}:
        raise ValueError("runtime.default_mode must be 'mock' or 'real'")
    if runtime.projection_warning_min_usd >= runtime.projection_warning_max_usd:
        raise ValueError("runtime.projection_warning_min_usd must be less than projection_warning_max_usd")
    if not runtime.real_required_env:
        raise ValueError("runtime.real_required_env must not be empty")
    if any(not env.strip() for env in runtime.real_required_env):
        raise ValueError("runtime.real_required_env contains empty environment variable name")
    if runtime.real_poll_interval_seconds <= 0:
        raise ValueError("runtime.real_poll_interval_seconds must be > 0")
    if runtime.real_poll_timeout_seconds <= 0:
        raise ValueError("runtime.real_poll_timeout_seconds must be > 0")


def _validate_evaluation(evaluation: EvaluationConfig) -> None:
    if evaluation.prompt_limit <= 0:
        raise ValueError("evaluation.prompt_limit must be > 0")


def _validate_campaign(campaign: CampaignConfig) -> None:
    if not campaign.seeds:
        raise ValueError("campaign.seeds must not be empty")
    if campaign.min_runs <= 0:
        raise ValueError("campaign.min_runs must be > 0")
    if campaign.max_runs <= 0:
        raise ValueError("campaign.max_runs must be > 0")
    if campaign.min_runs > campaign.max_runs:
        raise ValueError("campaign.min_runs cannot be greater than campaign.max_runs")
    if campaign.max_runs > len(campaign.seeds):
        raise ValueError("campaign.max_runs cannot exceed number of campaign.seeds")
    if campaign.bootstrap_reps <= 0:
        raise ValueError("campaign.bootstrap_reps must be > 0")
    if campaign.early_stop_threshold < 0:
        raise ValueError("campaign.early_stop_threshold must be >= 0")


def _resolve_path(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def load_config(path: Path | str = Path("config/default.toml")) -> ProjectConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text())
    project = data["project"]
    budget_raw = data["budget"]
    models = data["models"]

    budget = BudgetConfig(
        target_cap_usd=float(budget_raw["target_cap_usd"]),
        hard_cap_usd=float(budget_raw["hard_cap_usd"]),
        stage_budgets_usd={k: float(v) for k, v in data["stage_budgets_usd"].items()},
    )
    _validate_budget(budget)
    token_rates = {k: float(v) for k, v in data["token_rates_per_million"].items()}
    runtime_raw = data.get("runtime", {})
    runtime = RuntimeConfig(
        default_mode=str(runtime_raw.get("default_mode", "mock")),
        projection_warning_min_usd=float(runtime_raw.get("projection_warning_min_usd", 20.0)),
        projection_warning_max_usd=float(runtime_raw.get("projection_warning_max_usd", 30.0)),
        real_required_env=tuple(
            str(name)
            for name in runtime_raw.get(
                "real_required_env",
                ["TINKER_API_KEY", "TINKER_BASE_URL"],
            )
        ),
        real_poll_interval_seconds=int(runtime_raw.get("real_poll_interval_seconds", 15)),
        real_poll_timeout_seconds=int(runtime_raw.get("real_poll_timeout_seconds", 3600)),
    )
    _validate_runtime(runtime)
    default_prompt_file = Path(__file__).resolve().parent / "fixtures" / "real_eval_prompts_150.jsonl"
    evaluation_raw = data.get("evaluation", {})
    evaluation = EvaluationConfig(
        prompt_file=_resolve_path(config_path, str(evaluation_raw.get("prompt_file", default_prompt_file))),
        prompt_limit=int(evaluation_raw.get("prompt_limit", 150)),
    )
    _validate_evaluation(evaluation)
    campaign_raw = data.get("campaign", {})
    campaign = CampaignConfig(
        seeds=tuple(int(seed) for seed in campaign_raw.get("seeds", [17, 29, 43])),
        min_runs=int(campaign_raw.get("min_runs", 2)),
        max_runs=int(campaign_raw.get("max_runs", 3)),
        bootstrap_reps=int(campaign_raw.get("bootstrap_reps", 5000)),
        early_stop_threshold=float(campaign_raw.get("early_stop_threshold", 0.03)),
    )
    _validate_campaign(campaign)

    return ProjectConfig(
        name=str(project["name"]),
        seed=int(project["seed"]),
        teacher_model=str(models["teacher"]),
        student_model=str(models["student"]),
        baseline_model=str(models["baseline"]),
        token_rates_per_million=token_rates,
        token_caps=_token_caps_by_stage(data["token_caps"]),
        budget=budget,
        runtime=runtime,
        evaluation=evaluation,
        campaign=campaign,
    )
