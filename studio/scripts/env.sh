#!/usr/bin/env bash
# Shared configuration. Explicit settings take precedence over discovery.
STUDIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$STUDIO_DIR/.." && pwd)"
export STUDIO_DIR REPO_ROOT

if [[ -z "${STUDIO_PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    STUDIO_PYTHON="$REPO_ROOT/.venv/bin/python"
  elif [[ -f "$STUDIO_DIR/backend/workspace/envs/legommenders/.studio-ready" ]]; then
    STUDIO_PYTHON="$STUDIO_DIR/backend/workspace/envs/legommenders/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    STUDIO_PYTHON="$CONDA_PREFIX/bin/python"
  else
    STUDIO_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$STUDIO_PYTHON" ]] || ! "$STUDIO_PYTHON" -c 'import sys' >/dev/null 2>&1; then
  echo 'Python is unavailable. Install Python 3.11 or set STUDIO_PYTHON.' >&2
  return 1
fi
export STUDIO_PYTHON

if [[ -n "${STUDIO_NODE_BIN:-}" ]]; then
  export PATH="$STUDIO_NODE_BIN:$PATH"
fi
if ! command -v node >/dev/null 2>&1; then
  for candidate in "$HOME/.legommenders-tools/node20/bin" \
    "$(dirname "$REPO_ROOT")/.legommenders-tools/node20/bin"; do
    if [[ -x "$candidate/node" ]]; then
      export PATH="$candidate:$PATH"
      break
    fi
  done
fi

export STUDIO_HOST="${STUDIO_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export STUDIO_DEV="${STUDIO_DEV:-0}"
export STUDIO_API_URL="${STUDIO_API_URL:-http://127.0.0.1:$BACKEND_PORT}"
export PYTHONDONTWRITEBYTECODE=1
