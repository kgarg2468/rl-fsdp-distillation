from pathlib import Path

import pytest

from inference_projects.config import load_config


def test_invalid_budget_caps_rejected(tmp_path: Path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text(
        Path("config/default.toml")
        .read_text()
        .replace("target_cap_usd = 25.0", "target_cap_usd = 31.0")
    )
    with pytest.raises(ValueError):
        load_config(cfg)


def test_missing_stage_budget_rejected(tmp_path: Path):
    cfg = tmp_path / "bad_stage.toml"
    cfg.write_text(Path("config/default.toml").read_text().replace("eval = 3.0\n", ""))
    with pytest.raises(ValueError):
        load_config(cfg)
