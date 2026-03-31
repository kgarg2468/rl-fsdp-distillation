from pathlib import Path

from inference_projects.pipeline import run_pipeline_command


def test_golden_report_fixture_matches_mock_report(tmp_path: Path):
    state_dir = tmp_path / "state"
    run_pipeline_command("all", mode="mock", state_dir=state_dir)

    actual = (state_dir / "artifacts/reports/eval_report.md").read_text().strip()
    expected = Path("tests/fixtures/golden_eval_report.md").read_text().strip()
    assert actual == expected
