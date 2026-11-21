"""Studio-wide paths and runtime constants.

The studio is deliberately kept *outside* the Legommenders core packages so
that nothing in ``model/``, ``loader/`` or ``trainer.py`` has to be touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT: Path = Path(__file__).resolve().parent
STUDIO_ROOT: Path = BACKEND_ROOT.parent
REPO_ROOT: Path = STUDIO_ROOT.parent

# Make the original Legommenders packages importable (loader, model, utils,
# trainer, tester, ...).  This is the only coupling to the core project.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKSPACE: Path = BACKEND_ROOT / "workspace"
RUNS_DIR: Path = WORKSPACE / "runs"
GENERATED_DIR: Path = WORKSPACE / "generated"
GRAPHS_DIR: Path = WORKSPACE / "graphs"

CONFIG_DIR: Path = REPO_ROOT / "config"
MODEL_CONFIG_DIR: Path = CONFIG_DIR / "model"
DATA_CONFIG_DIR: Path = CONFIG_DIR / "data"
RECBENCH_CONFIG_DIR: Path = CONFIG_DIR / "recbench"
EMBED_CONFIG_DIR: Path = CONFIG_DIR / "embed"
EXP_CONFIG_DIR: Path = CONFIG_DIR / "exp"

for _directory in (WORKSPACE, RUNS_DIR, GENERATED_DIR, GRAPHS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# Client-supplied paths are never used verbatim: they are always resolved
# against these server-controlled allow-lists.
TRAINER_SCRIPT: Path = REPO_ROOT / "trainer.py"
TESTER_SCRIPT: Path = REPO_ROOT / "tester.py"
DEFAULT_EXP_CONFIG: Path = EXP_CONFIG_DIR / "default.yaml"
DEFAULT_EMBED_CONFIG: Path = EMBED_CONFIG_DIR / "null.yaml"

TASKS = ("matching", "ranking")
GRAPH_SCHEMA_VERSION = "1.0"
