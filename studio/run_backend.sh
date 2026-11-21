#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/scripts/env.sh"
cd "$REPO_ROOT"
echo "Legoizer backend: http://${STUDIO_HOST}:${BACKEND_PORT}"
echo "Python: $STUDIO_PYTHON"
if [[ -n "${STUDIO_HTTPS_PORT:-}" ]]; then
  : "${STUDIO_TLS_CERT:?Set STUDIO_TLS_CERT to the certificate PEM file}"
  : "${STUDIO_TLS_KEY:?Set STUDIO_TLS_KEY to the private key PEM file}"
  exec "$STUDIO_PYTHON" -m hypercorn \
    --bind "$STUDIO_HOST:$STUDIO_HTTPS_PORT" --insecure-bind "$STUDIO_HOST:$BACKEND_PORT" \
    --certfile "$STUDIO_TLS_CERT" --keyfile "$STUDIO_TLS_KEY" \
    --workers 1 studio.backend.app:app
fi
exec "$STUDIO_PYTHON" -m uvicorn studio.backend.app:app \
  --host "$STUDIO_HOST" --port "$BACKEND_PORT" --app-dir "$REPO_ROOT"
