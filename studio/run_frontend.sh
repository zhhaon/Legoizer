#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/scripts/env.sh"
cd "$STUDIO_DIR/frontend"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo 'Node.js and npm are required. See README.md or set STUDIO_NODE_BIN.' >&2
  exit 1
fi
if [[ ! -d node_modules ]]; then
  npm ci --registry="${NPM_REGISTRY:-https://registry.npmjs.org}"
fi
echo "Legoizer frontend: http://${STUDIO_HOST}:${FRONTEND_PORT}"
exec npm run dev -- --host "$STUDIO_HOST" --port "$FRONTEND_PORT" --strictPort
