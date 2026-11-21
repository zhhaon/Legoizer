#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/scripts/env.sh"
cd "$REPO_ROOT"
exec "$STUDIO_PYTHON" "$STUDIO_DIR/scripts/doctor.py" "$@"
