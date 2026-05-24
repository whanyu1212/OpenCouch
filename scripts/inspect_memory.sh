#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${OPENCOUCH_PERSISTENCE_BACKEND:-postgres}"

for arg in "$@"; do
  case "$arg" in
    --backend=sqlite)
      BACKEND="sqlite"
      ;;
    --backend=postgres)
      BACKEND="postgres"
      ;;
  esac
done

if [ "$BACKEND" != "sqlite" ]; then
  docker compose -f "$REPO_ROOT/compose.yml" up -d postgres --wait
  export OPENCOUCH_PERSISTENCE_BACKEND="${OPENCOUCH_PERSISTENCE_BACKEND:-postgres}"
  export OPENCOUCH_MEMORY_DATABASE_URL="${OPENCOUCH_MEMORY_DATABASE_URL:-postgresql://opencouch:opencouch@localhost:5432/opencouch}"
fi

cd "$REPO_ROOT/apps/backend"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python ../../scripts/inspect_memory.py "$@"
fi

exec uv run python ../../scripts/inspect_memory.py "$@"
