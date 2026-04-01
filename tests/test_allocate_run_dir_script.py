from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys


def _run_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    return subprocess.run(
        [sys.executable, "scripts/allocate_run_dir.py", *args],
        cwd=Path.cwd(),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_allocate_run_dir_script_creates_incremental_dirs(tmp_path: Path):
    root = tmp_path / "runs"
    first = _run_script(tmp_path, "--root", str(root))
    second = _run_script(tmp_path, "--root", str(root))
    assert first.returncode == 0
    assert second.returncode == 0
    first_path = Path(first.stdout.strip())
    second_path = Path(second.stdout.strip())
    assert first_path.name == "run-001"
    assert second_path.name == "run-002"
    assert first_path.exists()
    assert second_path.exists()


def test_allocate_run_dir_script_dry_run_does_not_create(tmp_path: Path):
    root = tmp_path / "runs"
    dry = _run_script(tmp_path, "--root", str(root), "--dry-run")
    assert dry.returncode == 0
    expected = Path(dry.stdout.strip())
    assert expected.name == "run-001"
    assert not expected.exists()
