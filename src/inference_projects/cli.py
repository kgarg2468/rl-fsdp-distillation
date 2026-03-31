from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_projects.pipeline import run_pipeline_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL + FSDP + Distillation pipeline")
    parser.add_argument(
        "command",
        choices=["rl", "fsdp", "distill", "eval", "report", "all", "smoke", "preflight", "dryrun"],
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline_command(
        args.command,
        mode=args.mode,
        config_path=Path(args.config),
        state_dir=Path(args.state_dir),
    )
    if result is not None:
        print(json.dumps(result, indent=2))
        if args.command == "preflight" and not result.get("ok", False):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
