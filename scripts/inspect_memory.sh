#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPECT_BACKEND_VALUE=0
for arg in "$@"; do
  if [ "$EXPECT_BACKEND_VALUE" -eq 1 ]; then
    case "$arg" in
      auto|postgres) ;;
      *)
        echo "Unsupported memory backend: $arg. Use auto or postgres." >&2
        exit 2
        ;;
    esac
    EXPECT_BACKEND_VALUE=0
    continue
  fi
  case "$arg" in
    --backend)
      EXPECT_BACKEND_VALUE=1
      ;;
    --backend=auto|--backend=postgres)
      ;;
    --backend=*)
      echo "Unsupported memory backend: ${arg#--backend=}. Use auto or postgres." >&2
      exit 2
      ;;
    --sqlite-path|--sqlite-path=*)
      echo "SQLite memory tooling has been removed; use Postgres." >&2
      exit 2
      ;;
  esac
done

if [ "$EXPECT_BACKEND_VALUE" -eq 1 ]; then
  echo "--backend requires auto or postgres." >&2
  exit 2
fi

docker compose -f "$REPO_ROOT/compose.yml" up -d postgres --wait
export OPENCOUCH_PERSISTENCE_BACKEND="postgres"
export OPENCOUCH_MEMORY_DATABASE_URL="${OPENCOUCH_MEMORY_DATABASE_URL:-postgresql://opencouch:opencouch@localhost:5432/opencouch}"

cd "$REPO_ROOT/apps/backend"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python ../../scripts/inspect_memory.py "$@"
fi

exec uv run python ../../scripts/inspect_memory.py "$@"
