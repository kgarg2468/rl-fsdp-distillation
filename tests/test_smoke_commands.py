from pathlib import Path

from inference_projects.pipeline import run_pipeline_command


def test_stage_by_stage_smoke(tmp_path: Path):
    state_dir = tmp_path / "state"
    run_pipeline_command("rl", state_dir=state_dir)
    run_pipeline_command("fsdp", state_dir=state_dir)
    run_pipeline_command("distill", state_dir=state_dir)
    run_pipeline_command("eval", state_dir=state_dir)
    run_pipeline_command("report", state_dir=state_dir)

    assert (state_dir / "artifacts/checkpoints/teacher/best_checkpoint.json").exists()
    assert (state_dir / "artifacts/checkpoints/student/best_checkpoint.json").exists()
    assert (state_dir / "artifacts/eval/eval_metrics.json").exists()
    assert (state_dir / "artifacts/reports/eval_report.md").exists()
