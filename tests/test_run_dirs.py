from pathlib import Path

from inference_projects.run_dirs import allocate_next_run_dir, latest_run_ledger, next_run_dir


def test_next_run_dir_uses_incremental_suffix(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "run-001").mkdir()
    (runs_root / "run-003").mkdir()
    (runs_root / "notes").mkdir()

    target = next_run_dir(runs_root)
    assert target.name == "run-004"


def test_allocate_next_run_dir_creates_directory(tmp_path: Path):
    runs_root = tmp_path / "runs"
    first = allocate_next_run_dir(runs_root)
    second = allocate_next_run_dir(runs_root)

    assert first.name == "run-001"
    assert second.name == "run-002"
    assert first.exists()
    assert second.exists()


def test_latest_run_ledger_prefers_latest_numbered_run(tmp_path: Path):
    runs_root = tmp_path / "runs"
    (runs_root / "run-001" / "artifacts").mkdir(parents=True)
    first_ledger = runs_root / "run-001" / "artifacts" / "ledger.json"
    first_ledger.write_text("{\"total_spend_usd\":0.1}\n")
    (runs_root / "run-002" / "campaign" / "runs" / "seed-17" / "artifacts").mkdir(parents=True)
    campaign_ledger = runs_root / "run-002" / "campaign" / "runs" / "seed-17" / "artifacts" / "ledger.json"
    campaign_ledger.write_text("{\"total_spend_usd\":0.2}\n")

    latest = latest_run_ledger(runs_root)
    assert latest == campaign_ledger
