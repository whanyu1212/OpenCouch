#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="auto"
EXPECT_BACKEND_VALUE=0

for arg in "$@"; do
  if [ "$EXPECT_BACKEND_VALUE" -eq 1 ]; then
    BACKEND="$arg"
    EXPECT_BACKEND_VALUE=0
    continue
  fi
  case "$arg" in
    --backend)
      EXPECT_BACKEND_VALUE=1
      ;;
    --backend=*)
      BACKEND="${arg#--backend=}"
      ;;
  esac
done

if [ "$BACKEND" != "sqlite" ]; then
  docker compose -f "$REPO_ROOT/compose.yml" up -d postgres --wait
  export OPENCOUCH_PERSISTENCE_BACKEND="postgres"
  export OPENCOUCH_MEMORY_DATABASE_URL="${OPENCOUCH_MEMORY_DATABASE_URL:-postgresql://opencouch:opencouch@localhost:5432/opencouch}"
fi

cd "$REPO_ROOT/apps/backend"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python ../../scripts/clear_memory.py "$@"
fi

exec uv run python ../../scripts/clear_memory.py "$@"
