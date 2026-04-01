#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference_projects.run_dirs import allocate_next_run_dir, latest_run_ledger, next_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Allocate next run directory under runs root.")
    parser.add_argument("--root", default="runs", help="Parent directory containing run-### folders.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print next run path without creating it.",
    )
    parser.add_argument(
        "--latest-ledger",
        action="store_true",
        help="Print latest ledger path under run folders instead of allocating a new run directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root)
    if args.latest_ledger:
        ledger = latest_run_ledger(root)
        if ledger is None:
            raise SystemExit(1)
        print(str(ledger.resolve()))
        return
    if args.dry_run:
        print(str(next_run_dir(root).resolve()))
        return
    allocated = allocate_next_run_dir(root)
    print(str(allocated.resolve()))


if __name__ == "__main__":
    main()
