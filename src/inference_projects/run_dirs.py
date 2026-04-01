from __future__ import annotations

from pathlib import Path
import re

RUN_DIR_RE = re.compile(r"^run-(\d{3})$")


def next_run_dir(root: Path) -> Path:
    root = root.resolve()
    highest = 0
    if root.exists():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            match = RUN_DIR_RE.match(child.name)
            if match is None:
                continue
            highest = max(highest, int(match.group(1)))
    return root / f"run-{highest + 1:03d}"


def allocate_next_run_dir(root: Path) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate = next_run_dir(root)
    while True:
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            match = RUN_DIR_RE.match(candidate.name)
            next_id = int(match.group(1)) + 1 if match else 1
            candidate = root / f"run-{next_id:03d}"


def latest_run_ledger(root: Path) -> Path | None:
    root = root.resolve()
    if not root.exists():
        return None
    numbered_dirs: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = RUN_DIR_RE.match(child.name)
        if match is None:
            continue
        numbered_dirs.append((int(match.group(1)), child))
    for _, run_dir in sorted(numbered_dirs, key=lambda item: item[0], reverse=True):
        single_run = run_dir / "artifacts/ledger.json"
        if single_run.exists():
            return single_run
        campaign_root = run_dir / "campaign" / "runs"
        if campaign_root.exists():
            seed_ledgers = sorted(campaign_root.glob("seed-*/artifacts/ledger.json"))
            if seed_ledgers:
                return seed_ledgers[-1]
    return None
