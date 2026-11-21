#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/scripts/env.sh"
exec "$STUDIO_PYTHON" "$STUDIO_DIR/scripts/services.py" run
