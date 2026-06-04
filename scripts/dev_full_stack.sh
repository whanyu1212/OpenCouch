#!/usr/bin/env bash
set -euo pipefail

echo "Starting full OpenCouch Compose stack"
echo "  Web:      http://localhost:3000"
echo "  API:      http://localhost:8080/api"
echo "  Health:   http://localhost:8080/api/health"
echo "  Postgres: postgresql://opencouch:opencouch@localhost:5432/opencouch"
echo
echo "If Docker build context fails on an external macOS volume, try:"
echo "  xattr -cr ."
echo

docker compose --profile web up
