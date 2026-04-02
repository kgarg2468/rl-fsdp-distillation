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
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from command run-state when available (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and run from scratch.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Heartbeat interval while stages are in progress.",
    )
    parser.add_argument(
        "--progress-timeout-seconds",
        type=float,
        default=None,
        help="Mark stage as stalled if elapsed time exceeds this value.",
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
        resume=args.resume,
        heartbeat_seconds=args.heartbeat_seconds,
        progress_timeout_seconds=args.progress_timeout_seconds,
    )
    if result is not None:
        print(json.dumps(result, indent=2))
        if args.command == "preflight" and not result.get("ok", False):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
