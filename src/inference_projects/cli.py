from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_projects.pipeline import run_pipeline_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL + FSDP + Distillation pipeline")
    parser.add_argument(
        "command",
        choices=["rl", "fsdp", "distill", "eval", "report", "all", "smoke", "preflight", "dryrun", "campaign", "tune"],
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "real"],
        default=None,
        help="Runtime mode (default: config.runtime.default_mode).",
    )
    parser.add_argument("--config", default="config/default.toml", help="Path to TOML config file.")
    parser.add_argument(
        "--state-dir",
        default=".",
        help="Directory where artifacts are written (default: current project root).",
    )
    parser.add_argument(
        "--prior-ledger",
        default=None,
        help="Optional path to a previous ledger.json used to seed prior spend for campaign budget checks.",
    )
    parser.add_argument(
        "--project-hard-cap-usd",
        type=float,
        default=35.0,
        help="Project-level hard cap for campaign runs, including prior spend.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline_command(
        args.command,
        mode=args.mode,
        config_path=Path(args.config),
        state_dir=Path(args.state_dir),
        prior_ledger=Path(args.prior_ledger) if args.prior_ledger else None,
        project_hard_cap_usd=float(args.project_hard_cap_usd),
    )
    if result is not None:
        print(json.dumps(result, indent=2))
        if args.command == "preflight" and not result.get("ok", False):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
