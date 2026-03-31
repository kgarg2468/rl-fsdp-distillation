from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from inference_projects import budget
from inference_projects.config import ProjectConfig, REQUIRED_STAGES


class SetupError(RuntimeError):
    """Raised when environment or runtime setup is incomplete."""


@dataclass(frozen=True)
class PreflightResult:
    mode: str
    ok: bool
    projected_total_usd: float
    errors: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "projected_total_usd": round(self.projected_total_usd, 4),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def run_preflight(
    *,
    mode: str,
    cfg: ProjectConfig,
    state_dir: Path | str = Path("."),
    check_state_dir: bool = True,
) -> PreflightResult:
    errors: list[str] = []
    warnings: list[str] = []

    projected_total = budget.projected_total_cost_usd(cfg)

    for stage in REQUIRED_STAGES:
        stage_cost = budget.projected_stage_cost_usd(stage, cfg)
        try:
            budget.ensure_within_stage_budget(stage, stage_cost, cfg)
        except budget.BudgetExceededError as exc:
            errors.append(str(exc))

    running_total = 0.0
    for stage in REQUIRED_STAGES:
        stage_cost = budget.projected_stage_cost_usd(stage, cfg)
        try:
            budget.ensure_within_hard_cap(current_total=running_total, incoming_cost=stage_cost, cfg=cfg)
        except budget.BudgetExceededError as exc:
            errors.append(str(exc))
        running_total += stage_cost

    if projected_total < cfg.runtime.projection_warning_min_usd or projected_total > cfg.runtime.projection_warning_max_usd:
        warnings.append(
            "Projected total spend is outside target warning band "
            f"${cfg.runtime.projection_warning_min_usd:.2f}-${cfg.runtime.projection_warning_max_usd:.2f}."
        )

    if check_state_dir:
        _check_state_dir_permissions(Path(state_dir), errors)

    if mode == "real":
        for key in cfg.runtime.real_required_env:
            if not os.getenv(key):
                errors.append(f"Missing required environment variable for real mode: {key}")

    if mode not in {"mock", "real"}:
        errors.append(f"Unsupported mode: {mode}")

    return PreflightResult(
        mode=mode,
        ok=not errors,
        projected_total_usd=projected_total,
        errors=errors,
        warnings=warnings,
    )


def ensure_preflight_ready(*, mode: str, cfg: ProjectConfig, state_dir: Path | str = Path(".")) -> PreflightResult:
    result = run_preflight(mode=mode, cfg=cfg, state_dir=state_dir)
    if not result.ok:
        joined = "\n".join(result.errors)
        raise SetupError(f"Preflight failed for mode '{mode}':\n{joined}")
    return result


def _check_state_dir_permissions(state_dir: Path, errors: list[str]) -> None:
    probe_dir = state_dir / "artifacts"
    probe_file = probe_dir / ".preflight_write_test"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_file.write_text("ok\n")
        probe_file.unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"State directory is not writable: {state_dir} ({exc})")
