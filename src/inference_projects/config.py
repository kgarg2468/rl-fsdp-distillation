from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from inference_projects.pricing import TokenUsage

REQUIRED_STAGES = ("rl", "teacher_ft", "distill", "eval")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    teacher_model: str
    student_model: str
    baseline_model: str
    token_rates_per_million: dict[str, float]
    token_caps: dict[str, TokenUsage]
    runtime: "RuntimeConfig"
    evaluation: "EvaluationConfig"
    distillation: "DistillationConfig"
    campaign: "CampaignConfig"
    tuning: "TuningConfig"


@dataclass(frozen=True)
class RuntimeConfig:
    default_mode: str
    projection_warning_min_usd: float
    projection_warning_max_usd: float
    real_required_env: tuple[str, ...]
    real_poll_interval_seconds: int
    real_poll_timeout_seconds: int
    retry_max_connections: int
    retry_progress_timeout_seconds: float
    retry_delay_base_seconds: float
    retry_delay_max_seconds: float
    retry_jitter_factor: float
    retry_enabled: bool
    max_consecutive_failures: int


@dataclass(frozen=True)
class EvaluationConfig:
    prompt_file: Path
    prompt_limit: int
    max_concurrency: int
    batch_size: int
    max_tokens_eval: int
    eval_temperature: float
    eval_stop_tokens: tuple[str, ...]
    eval_max_tokens_candidates: tuple[int, ...]
    teacher_integrity_refusal_threshold: float
    teacher_integrity_min_score: float
    teacher_integrity_numeric_parse_threshold: float


@dataclass(frozen=True)
class DistillationConfig:
    training_prompt_limit: int
    teacher_prompt_template: str
    filter_profile: str
    hard_example_ratio: float
    kd_alpha: float
    kd_temperature: float
    learning_rate: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    lora_rank: int
    context_length: int
    grad_clip: float
    weight_decay: float


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    min_runs: int
    max_runs: int
    bootstrap_reps: int
    early_stop_threshold: float
    strict_run_cap: int


@dataclass(frozen=True)
class TuningConfig:
    stage1_prompt_limit: int
    stage2_prompt_limit: int
    sweep_runs: int
    teacher_candidates: tuple[str, ...]
    promotion_top_k: int
    strict_run_cap: int


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
    if runtime.retry_max_connections <= 0:
        raise ValueError("runtime.retry_max_connections must be > 0")
    if runtime.retry_progress_timeout_seconds <= 0:
        raise ValueError("runtime.retry_progress_timeout_seconds must be > 0")
    if runtime.retry_delay_base_seconds <= 0:
        raise ValueError("runtime.retry_delay_base_seconds must be > 0")
    if runtime.retry_delay_max_seconds <= 0:
        raise ValueError("runtime.retry_delay_max_seconds must be > 0")
    if runtime.retry_delay_max_seconds < runtime.retry_delay_base_seconds:
        raise ValueError("runtime.retry_delay_max_seconds must be >= runtime.retry_delay_base_seconds")
    if not (0.0 <= runtime.retry_jitter_factor <= 1.0):
        raise ValueError("runtime.retry_jitter_factor must be in [0, 1]")
    if runtime.max_consecutive_failures <= 0:
        raise ValueError("runtime.max_consecutive_failures must be > 0")


def _validate_evaluation(evaluation: EvaluationConfig) -> None:
    if evaluation.prompt_limit <= 0:
        raise ValueError("evaluation.prompt_limit must be > 0")
    if evaluation.max_concurrency <= 0:
        raise ValueError("evaluation.max_concurrency must be > 0")
    if evaluation.batch_size <= 0:
        raise ValueError("evaluation.batch_size must be > 0")
    if evaluation.max_tokens_eval <= 0:
        raise ValueError("evaluation.max_tokens_eval must be > 0")
    if evaluation.eval_temperature < 0:
        raise ValueError("evaluation.eval_temperature must be >= 0")
    if not evaluation.eval_max_tokens_candidates:
        raise ValueError("evaluation.eval_max_tokens_candidates must not be empty")
    if any(value <= 0 for value in evaluation.eval_max_tokens_candidates):
        raise ValueError("evaluation.eval_max_tokens_candidates must contain only values > 0")
    if not (0.0 <= evaluation.teacher_integrity_refusal_threshold <= 1.0):
        raise ValueError("evaluation.teacher_integrity_refusal_threshold must be in [0, 1]")
    if not (0.0 <= evaluation.teacher_integrity_min_score <= 1.0):
        raise ValueError("evaluation.teacher_integrity_min_score must be in [0, 1]")
    if not (0.0 <= evaluation.teacher_integrity_numeric_parse_threshold <= 1.0):
        raise ValueError("evaluation.teacher_integrity_numeric_parse_threshold must be in [0, 1]")


def _validate_distillation(distillation: DistillationConfig) -> None:
    if distillation.training_prompt_limit <= 0:
        raise ValueError("distillation.training_prompt_limit must be > 0")
    if distillation.teacher_prompt_template not in {"raw", "numeric_strict"}:
        raise ValueError("distillation.teacher_prompt_template must be 'raw' or 'numeric_strict'")
    if distillation.filter_profile not in {"moderate", "strict"}:
        raise ValueError("distillation.filter_profile must be 'moderate' or 'strict'")
    if not (0.0 <= distillation.hard_example_ratio <= 1.0):
        raise ValueError("distillation.hard_example_ratio must be in [0, 1]")
    if not (0.0 <= distillation.kd_alpha <= 1.0):
        raise ValueError("distillation.kd_alpha must be in [0, 1]")
    if distillation.kd_temperature <= 0:
        raise ValueError("distillation.kd_temperature must be > 0")
    if distillation.learning_rate <= 0:
        raise ValueError("distillation.learning_rate must be > 0")
    if distillation.epochs <= 0:
        raise ValueError("distillation.epochs must be > 0")
    if distillation.batch_size <= 0:
        raise ValueError("distillation.batch_size must be > 0")
    if not (0.0 <= distillation.warmup_ratio <= 1.0):
        raise ValueError("distillation.warmup_ratio must be in [0, 1]")
    if distillation.lora_rank <= 0:
        raise ValueError("distillation.lora_rank must be > 0")
    if distillation.context_length <= 0:
        raise ValueError("distillation.context_length must be > 0")
    if distillation.grad_clip <= 0:
        raise ValueError("distillation.grad_clip must be > 0")
    if distillation.weight_decay < 0:
        raise ValueError("distillation.weight_decay must be >= 0")


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
    if campaign.strict_run_cap <= 0:
        raise ValueError("campaign.strict_run_cap must be > 0")


def _validate_tuning(tuning: TuningConfig) -> None:
    if tuning.stage1_prompt_limit <= 0:
        raise ValueError("tuning.stage1_prompt_limit must be > 0")
    if tuning.stage2_prompt_limit <= 0:
        raise ValueError("tuning.stage2_prompt_limit must be > 0")
    if tuning.stage1_prompt_limit > tuning.stage2_prompt_limit:
        raise ValueError("tuning.stage1_prompt_limit cannot exceed tuning.stage2_prompt_limit")
    if tuning.sweep_runs <= 0:
        raise ValueError("tuning.sweep_runs must be > 0")
    if not tuning.teacher_candidates:
        raise ValueError("tuning.teacher_candidates must not be empty")
    if tuning.promotion_top_k <= 0:
        raise ValueError("tuning.promotion_top_k must be > 0")
    if tuning.promotion_top_k > tuning.sweep_runs:
        raise ValueError("tuning.promotion_top_k cannot exceed tuning.sweep_runs")
    if tuning.strict_run_cap <= 0:
        raise ValueError("tuning.strict_run_cap must be > 0")


def _resolve_path(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    relative_to_config = (config_path.parent / path).resolve()
    if relative_to_config.exists():
        return relative_to_config
    return path.resolve()


def load_config(path: Path | str = Path("config/default.toml")) -> ProjectConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text())
    project = data["project"]
    models = data["models"]
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
        retry_max_connections=int(runtime_raw.get("retry_max_connections", 1000)),
        retry_progress_timeout_seconds=float(runtime_raw.get("retry_progress_timeout_seconds", 7200)),
        retry_delay_base_seconds=float(runtime_raw.get("retry_delay_base_seconds", 0.5)),
        retry_delay_max_seconds=float(runtime_raw.get("retry_delay_max_seconds", 10.0)),
        retry_jitter_factor=float(runtime_raw.get("retry_jitter_factor", 0.25)),
        retry_enabled=bool(runtime_raw.get("retry_enabled", True)),
        max_consecutive_failures=int(runtime_raw.get("max_consecutive_failures", 5)),
    )
    _validate_runtime(runtime)
    teacher_model = str(models["teacher"])
    student_model = str(models["student"])
    baseline_model = str(models["baseline"])

    default_prompt_file = Path(__file__).resolve().parent / "fixtures" / "real_eval_prompts_150.jsonl"
    evaluation_raw = data.get("evaluation", {})
    evaluation = EvaluationConfig(
        prompt_file=_resolve_path(config_path, str(evaluation_raw.get("prompt_file", default_prompt_file))),
        prompt_limit=int(evaluation_raw.get("prompt_limit", 150)),
        max_concurrency=int(evaluation_raw.get("max_concurrency", 6)),
        batch_size=int(evaluation_raw.get("batch_size", 24)),
        max_tokens_eval=int(evaluation_raw.get("max_tokens_eval", 48)),
        eval_temperature=float(evaluation_raw.get("eval_temperature", 0.0)),
        eval_stop_tokens=tuple(str(token) for token in evaluation_raw.get("eval_stop_tokens", [])),
        eval_max_tokens_candidates=tuple(
            int(value) for value in evaluation_raw.get("eval_max_tokens_candidates", [48, 96])
        ),
        teacher_integrity_refusal_threshold=float(
            evaluation_raw.get("teacher_integrity_refusal_threshold", 0.30)
        ),
        teacher_integrity_min_score=float(evaluation_raw.get("teacher_integrity_min_score", 0.10)),
        teacher_integrity_numeric_parse_threshold=float(
            evaluation_raw.get("teacher_integrity_numeric_parse_threshold", 0.80)
        ),
    )
    _validate_evaluation(evaluation)

    distillation_raw = data.get("distillation", {})
    distillation = DistillationConfig(
        training_prompt_limit=int(distillation_raw.get("training_prompt_limit", 150)),
        teacher_prompt_template=str(distillation_raw.get("teacher_prompt_template", "raw")),
        filter_profile=str(distillation_raw.get("filter_profile", "moderate")),
        hard_example_ratio=float(distillation_raw.get("hard_example_ratio", 0.4)),
        kd_alpha=float(distillation_raw.get("kd_alpha", 0.5)),
        kd_temperature=float(distillation_raw.get("kd_temperature", 2.0)),
        learning_rate=float(distillation_raw.get("learning_rate", 0.00002)),
        epochs=int(distillation_raw.get("epochs", 2)),
        batch_size=int(distillation_raw.get("batch_size", 8)),
        warmup_ratio=float(distillation_raw.get("warmup_ratio", 0.1)),
        lora_rank=int(distillation_raw.get("lora_rank", 8)),
        context_length=int(distillation_raw.get("context_length", 4096)),
        grad_clip=float(distillation_raw.get("grad_clip", 1.0)),
        weight_decay=float(distillation_raw.get("weight_decay", 0.0)),
    )
    _validate_distillation(distillation)

    campaign_raw = data.get("campaign", {})
    campaign = CampaignConfig(
        seeds=tuple(int(seed) for seed in campaign_raw.get("seeds", [17, 29, 43])),
        min_runs=int(campaign_raw.get("min_runs", 2)),
        max_runs=int(campaign_raw.get("max_runs", 3)),
        bootstrap_reps=int(campaign_raw.get("bootstrap_reps", 5000)),
        early_stop_threshold=float(campaign_raw.get("early_stop_threshold", 0.03)),
        strict_run_cap=int(campaign_raw.get("strict_run_cap", 16)),
    )
    _validate_campaign(campaign)

    tuning_raw = data.get("tuning", {})
    tuning = TuningConfig(
        stage1_prompt_limit=int(tuning_raw.get("stage1_prompt_limit", 30)),
        stage2_prompt_limit=int(tuning_raw.get("stage2_prompt_limit", 150)),
        sweep_runs=int(tuning_raw.get("sweep_runs", 16)),
        teacher_candidates=tuple(
            str(model_name)
            for model_name in tuning_raw.get(
                "teacher_candidates",
                [teacher_model],
            )
        ),
        promotion_top_k=int(tuning_raw.get("promotion_top_k", 2)),
        strict_run_cap=int(tuning_raw.get("strict_run_cap", 16)),
    )
    _validate_tuning(tuning)

    return ProjectConfig(
        name=str(project["name"]),
        seed=int(project["seed"]),
        teacher_model=teacher_model,
        student_model=student_model,
        baseline_model=baseline_model,
        token_rates_per_million=token_rates,
        token_caps=_token_caps_by_stage(data["token_caps"]),
        runtime=runtime,
        evaluation=evaluation,
        distillation=distillation,
        campaign=campaign,
        tuning=tuning,
    )
