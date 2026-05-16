#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/apps/backend"
PYTHON="$BACKEND_DIR/.venv/bin/python"
START_POSTGRES=1

usage() {
  cat <<'EOF'
Usage: scripts/voice_agent.sh [options] [livekit-agent-command...]

Starts the standalone LiveKit voice worker or a local console session.
If no command is supplied, runs worker mode:
  agent.voice.agent start

Worker mode waits for a browser/LiveKit room participant and does not use
your local microphone. For local mic dogfooding, run:
  scripts/voice_agent.sh --user-id dogfood console

Options:
  --user-id ID             Set OPENCOUCH_VOICE_USER_ID.
  --thread-id ID           Set OPENCOUCH_VOICE_THREAD_ID.
  --memory-mode MODE       Set OPENCOUCH_MEMORY_MODE. Common values: local, incognito.
  --backend BACKEND        Set OPENCOUCH_PERSISTENCE_BACKEND. Common values: postgres, sqlite.
  --database-url URL       Set OPENCOUCH_MEMORY_DATABASE_URL.
  --no-postgres            Do not start the compose Postgres service.
  -h, --help               Show this help.

Examples:
  scripts/voice_agent.sh --user-id dogfood console
  scripts/voice_agent.sh --user-id dogfood console --text
  scripts/voice_agent.sh --memory-mode incognito console
  scripts/voice_agent.sh --user-id dogfood start
  scripts/voice_agent.sh --no-postgres --backend sqlite console --text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-id)
      if [[ $# -lt 2 ]]; then
        echo "--user-id requires a value" >&2
        exit 2
      fi
      export OPENCOUCH_VOICE_USER_ID="$2"
      shift 2
      ;;
    --thread-id)
      if [[ $# -lt 2 ]]; then
        echo "--thread-id requires a value" >&2
        exit 2
      fi
      export OPENCOUCH_VOICE_THREAD_ID="$2"
      shift 2
      ;;
    --memory-mode)
      if [[ $# -lt 2 ]]; then
        echo "--memory-mode requires a value" >&2
        exit 2
      fi
      export OPENCOUCH_MEMORY_MODE="$2"
      shift 2
      ;;
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "--backend requires a value" >&2
        exit 2
      fi
      export OPENCOUCH_PERSISTENCE_BACKEND="$2"
      shift 2
      ;;
    --database-url)
      if [[ $# -lt 2 ]]; then
        echo "--database-url requires a value" >&2
        exit 2
      fi
      export OPENCOUCH_MEMORY_DATABASE_URL="$2"
      shift 2
      ;;
    --no-postgres)
      START_POSTGRES=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Backend virtualenv not found at $PYTHON" >&2
  echo "Run: cd apps/backend && uv sync --extra voice" >&2
  exit 1
fi

if [[ "$START_POSTGRES" -eq 1 ]]; then
  docker compose -f "$REPO_ROOT/compose.yml" up -d postgres --wait
fi

cd "$BACKEND_DIR"

if [[ $# -eq 0 ]]; then
  set -- start
fi

exec "$PYTHON" -m agent.voice.agent "$@"
