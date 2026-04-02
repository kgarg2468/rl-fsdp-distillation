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
            "--state-dir",
            "/tmp/state",
        ]
    )
    assert args.command == "campaign"
    assert args.mode == "real"
    assert args.state_dir == "/tmp/state"


def test_cli_supports_tune_command():
    parser = build_parser()
    args = parser.parse_args(["tune", "--mode", "mock", "--state-dir", "/tmp/state"])
    assert args.command == "tune"
    assert args.mode == "mock"


def test_cli_parses_reliability_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "campaign",
            "--state-dir",
            "/tmp/state",
            "--no-resume",
            "--heartbeat-seconds",
            "12",
            "--progress-timeout-seconds",
            "99",
        ]
    )
    assert args.resume is False
    assert args.heartbeat_seconds == 12
    assert args.progress_timeout_seconds == 99.0
