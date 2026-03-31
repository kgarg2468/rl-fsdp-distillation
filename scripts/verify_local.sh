#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

TMP_BASE="$(mktemp -d /tmp/inference-projects-verify.XXXXXX)"
trap 'rm -rf "$TMP_BASE"' EXIT

python -m inference_projects.style_checks
python -m compileall -q src tests
pytest -q
python -m inference_projects.cli preflight --mode mock --config config/default.toml --state-dir "$TMP_BASE/preflight" >/dev/null
python -m inference_projects.cli dryrun --mode mock --config config/default.toml --state-dir "$TMP_BASE/dryrun" >/dev/null
python -m inference_projects.cli smoke --mode mock --config config/default.toml --state-dir "$TMP_BASE/smoke" >/dev/null
