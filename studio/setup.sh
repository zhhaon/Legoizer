#!/usr/bin/env bash
# Install into .venv; do not download datasets or model checkpoints.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export TMPDIR="${STUDIO_INSTALL_TMPDIR:-$REPO_ROOT/studio/backend/workspace/install/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/studio/backend/workspace/install/pip-cache}"
export npm_config_cache="${npm_config_cache:-$REPO_ROOT/studio/backend/workspace/install/npm-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$npm_config_cache"
BOOTSTRAP_PYTHON="${STUDIO_BOOTSTRAP_PYTHON:-python3.11}"
TORCH_FLAVOR=cpu
INSTALL_DEV=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) BOOTSTRAP_PYTHON="${2:?Missing Python executable}"; shift 2 ;;
    --torch) TORCH_FLAVOR="${2:?Choose cpu or cu128}"; shift 2 ;;
    --dev) INSTALL_DEV=1; shift ;;
    --help|-h) echo 'Usage: bash studio/setup.sh [--python /path/to/python3.11] [--torch cpu|cu128] [--dev]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$TORCH_FLAVOR" in cpu|cu128) ;; *) echo 'Choose --torch cpu or --torch cu128.' >&2; exit 2 ;; esac
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This release supports Linux and WSL2. Run these commands inside WSL2 on Windows.' >&2
  exit 1
fi
for command_name in curl node npm; do
  if [[ "$command_name" == node && -n "${STUDIO_NODE_BIN:-}" ]]; then export PATH="$STUDIO_NODE_BIN:$PATH"; fi
  command -v "$command_name" >/dev/null || { echo "Missing $command_name. See README.md." >&2; exit 1; }
done
node -e 'if (Number(process.versions.node.split(".")[0]) < 20) process.exit(1)' || { echo 'Node.js 20 or later is required.' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  "$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
  "$BOOTSTRAP_PYTHON" -m venv .venv
fi
PYTHON="$REPO_ROOT/.venv/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Recreate .venv with Python 3.11"'
"$PYTHON" -m pip install 'torch==2.9.1' --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_FLAVOR}"
"$PYTHON" -m pip install -c constraints-python311.txt -r requirements.txt -r studio/backend/requirements.txt
if [[ "$INSTALL_DEV" == 1 ]]; then
  "$PYTHON" -m pip install -c constraints-python311.txt -r requirements-dev.txt
fi
"$PYTHON" -m pip check
(cd studio/frontend && npm ci --registry="${NPM_REGISTRY:-https://registry.npmjs.org}" && npm run build)
STUDIO_PYTHON="$PYTHON" bash studio/doctor.sh
echo 'Installation complete. Run: bash studio/run.sh'
