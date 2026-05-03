#!/bin/sh
set -eu

corepack enable
corepack prepare pnpm@10.33.0 --activate

if ! node -e 'require.resolve("next/package.json", { paths: ["/workspace/apps/web"] })' >/dev/null 2>&1; then
  echo "web dependencies missing or incomplete; installing into the Docker volume..."
  pnpm --dir apps/web install --frozen-lockfile
fi

exec pnpm --dir apps/web dev --hostname 0.0.0.0
