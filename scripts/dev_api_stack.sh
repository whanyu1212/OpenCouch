#!/usr/bin/env bash
set -euo pipefail

echo "Starting OpenCouch Postgres + API"
echo "  API:      http://localhost:8080/api"
echo "  Health:   http://localhost:8080/api/health"
echo "  Postgres: postgresql://opencouch:opencouch@localhost:5432/opencouch"
echo
echo "Run the frontend separately with:"
echo "  cd apps/web && NEXT_PUBLIC_API_URL=http://localhost:8080/api pnpm dev"
echo
echo "If Docker build context fails on an external macOS volume, try:"
echo "  xattr -cr ."
echo

docker compose up postgres api
