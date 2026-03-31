from pathlib import Path

from inference_projects.config import load_config
from inference_projects.preflight import run_preflight


def test_preflight_warns_when_projected_total_outside_window(tmp_path: Path):
    cfg_path = tmp_path / "warn.toml"
    text = Path("config/default.toml").read_text()
    text = text.replace("projection_warning_min_usd = 20.0", "projection_warning_min_usd = 22.0")
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)

    result = run_preflight(mode="mock", cfg=cfg)
    assert result.ok
    assert result.warnings


def test_preflight_reports_unsupported_mode():
    cfg = load_config()
    result = run_preflight(mode="invalid", cfg=cfg)
    assert not result.ok
    assert any("Unsupported mode" in msg for msg in result.errors)
