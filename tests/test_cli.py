from inference_projects.cli import build_parser


def test_cli_parses_config_and_state_dir():
    parser = build_parser()
    args = parser.parse_args(["all", "--mode", "mock", "--config", "config/default.toml", "--state-dir", "tmp"])
    assert args.command == "all"
    assert args.mode == "mock"
    assert args.config == "config/default.toml"
    assert args.state_dir == "tmp"


def test_cli_supports_preflight_command():
    parser = build_parser()
    args = parser.parse_args(["preflight", "--mode", "real"])
    assert args.command == "preflight"
    assert args.mode == "real"


def test_cli_parses_campaign_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "campaign",
            "--mode",
            "real",
            "--prior-ledger",
            "/tmp/prior-ledger.json",
            "--project-hard-cap-usd",
            "35.0",
            "--state-dir",
            "/tmp/state",
        ]
    )
    assert args.command == "campaign"
    assert args.mode == "real"
    assert args.prior_ledger == "/tmp/prior-ledger.json"
    assert args.project_hard_cap_usd == 35.0


def test_cli_supports_tune_command():
    parser = build_parser()
    args = parser.parse_args(["tune", "--mode", "mock", "--state-dir", "/tmp/state"])
    assert args.command == "tune"
    assert args.mode == "mock"
