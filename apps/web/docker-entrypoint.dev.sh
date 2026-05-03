#!/bin/sh
set -eu

corepack enable
corepack prepare pnpm@10.33.0 --activate

cd /workspace

deps_hash="$(sha256sum pnpm-lock.yaml pnpm-workspace.yaml apps/web/package.json | sha256sum | awk '{print $1}')"
deps_sentinel="/workspace/apps/web/node_modules/.docker-pnpm-deps.sha256"

deps_are_usable() {
  node <<'NODE' >/dev/null 2>&1
const { createRequire } = require("module");
const requireFromWeb = createRequire("/workspace/apps/web/package.json");

requireFromWeb("next/package.json");
requireFromWeb("@tailwindcss/postcss");
NODE
}

if ! deps_are_usable || [ ! -f "$deps_sentinel" ] || [ "$(cat "$deps_sentinel")" != "$deps_hash" ]; then
  echo "web dependencies missing, stale, or incomplete; installing into Docker volumes..."
  pnpm --dir apps/web install --frozen-lockfile
  if ! deps_are_usable; then
    echo "web dependency install completed, but native packages are still unavailable." >&2
    exit 1
  fi
  printf '%s\n' "$deps_hash" > "$deps_sentinel"
fi

exec pnpm --dir apps/web dev --hostname 0.0.0.0
