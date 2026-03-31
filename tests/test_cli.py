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
