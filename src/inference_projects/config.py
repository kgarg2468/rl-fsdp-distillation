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


@dataclass(frozen=True)
class RuntimeConfig:
    default_mode: str
    projection_warning_min_usd: float
    projection_warning_max_usd: float
    real_required_env: tuple[str, ...]
    real_poll_interval_seconds: int
    real_poll_timeout_seconds: int


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


def load_config(path: Path | str = Path("config/default.toml")) -> ProjectConfig:
    data = tomllib.loads(Path(path).read_text())
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
    )
