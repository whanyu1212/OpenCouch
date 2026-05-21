#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../apps/backend"
exec uv run python -m opencouch_tui "$@"
