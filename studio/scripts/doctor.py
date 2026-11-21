"""Check the runtime without downloading data, weights, or starting training."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    failures = []

    def report(ok, label, detail=""):
        print(f"{'OK' if ok else 'FAIL'}  {label}{': ' + str(detail) if detail else ''}")
        if not ok:
            failures.append(label)

    report(sys.platform == "linux", "Operating system", sys.platform)
    report(sys.version_info[:2] == (3, 11), "Python 3.11", sys.executable)
    for name in ("torch", "transformers", "unitok", "refconfig", "fastapi", "pydantic", "uvicorn", "hypercorn"):
        try:
            report(True, name, version(name))
        except PackageNotFoundError:
            report(False, name, "not installed; run bash studio/setup.sh")
    checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report(checked.returncode == 0, "Python dependency consistency", (checked.stdout + checked.stderr).strip())
    report(shutil.which("curl") is not None, "curl (resource downloads)")
    report(os.access(ROOT, os.W_OK), "Writable checkout")
    built = (ROOT / "studio/frontend/dist/index.html").is_file()
    report(built, "Frontend build", "run npm ci && npm run build in studio/frontend" if not built else "ready")
    node = shutil.which("node")
    if node:
        result = subprocess.run([node, "-p", "process.versions.node"], capture_output=True, text=True)
        value = result.stdout.strip()
        report(result.returncode == 0 and int(value.split(".")[0]) >= 20, "Node.js", value)
    else:
        print("INFO  Node.js is absent; only needed to install/build or develop the frontend.")
    try:
        import torch
        print(f"INFO  PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
        for index in range(torch.cuda.device_count()):
            print(f"INFO  GPU {index}: {torch.cuda.get_device_name(index)}")
        if "--require-cuda" in sys.argv:
            report(torch.cuda.is_available(), "CUDA required")
    except Exception as exc:
        report(False, "PyTorch import", str(exc))
    host, port = os.environ.get("STUDIO_HOST", "127.0.0.1"), int(os.environ.get("BACKEND_PORT", "8000"))
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        print(f"INFO  {host}:{port} is available.")
    except OSError:
        print(f"INFO  {host}:{port} is in use; stop your running instance or select BACKEND_PORT.")
    print("INFO  .data is optional; missing datasets are downloaded when requested.")
    print(f"INFO  Optional local dataset overrides: {'present' if (ROOT / '.data').is_file() else 'absent (automatic download)'}")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
