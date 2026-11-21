"""Allow ``python -m studio.backend`` to launch the API server."""
from __future__ import annotations

import os

import uvicorn

from studio.backend.config import REPO_ROOT


def main() -> None:
    uvicorn.run(
        "studio.backend.app:app",
        host=os.environ.get("STUDIO_HOST", "127.0.0.1"),
        port=int(os.environ.get("BACKEND_PORT", "8000")),
        app_dir=str(REPO_ROOT),
    )


if __name__ == "__main__":
    main()
