#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker compose -f "$REPO_ROOT/compose.yml" up -d postgres --wait

cd "$REPO_ROOT/apps/backend"
exec uv run python -m opencouch_tui "$@"
