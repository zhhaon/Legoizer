"""Run this checkout's services in the foreground; stop only its own manager."""
from __future__ import annotations

import fcntl
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "studio/backend/workspace/runtime"
STATE = RUNTIME / "services.json"


def identity(pid: int) -> str | None:
    try:
        # starttime protects against PID reuse; comm can contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def stop() -> int:
    if not STATE.is_file():
        print("No managed Legoizer service is running in this checkout.")
        return 0
    state = json.loads(STATE.read_text())
    pid = state["pid"]
    if identity(pid) != state["starttime"]:
        print("The previous Legoizer manager has already stopped.")
        return 0
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if os.fsencode(str(Path(__file__).resolve())) not in command:
        raise RuntimeError("Manager identity did not match; no process was signalled.")
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if identity(pid) != state["starttime"]:
            print("Stopped this checkout's Legoizer services.")
            return 0
        time.sleep(0.2)
    print("Manager is still stopping; inspect studio/logs/.", file=sys.stderr)
    return 1


class StopRequested(Exception):
    pass


def run() -> int:
    os.chdir(ROOT)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "services.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Legoizer is already running in this checkout.", file=sys.stderr)
        return 1
    dev = os.environ.get("STUDIO_DEV", "0") == "1"
    if not dev and not (ROOT / "studio/frontend/dist/index.html").is_file():
        raise RuntimeError("Frontend is not built. Run bash studio/setup.sh first.")
    host = os.environ.get("STUDIO_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    # Never mistake a different service's health endpoint for our child.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    children = []
    streams = []
    logs = ROOT / "studio/logs"
    logs.mkdir(exist_ok=True)
    state = {"pid": os.getpid(), "starttime": identity(os.getpid())}
    STATE.write_text(json.dumps(state))

    def interrupted(_signum, _frame):
        raise StopRequested()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(name):
        stream = (logs / f"{name}.log").open("w")
        streams.append(stream)
        process = subprocess.Popen(
            ["bash", str(ROOT / "studio" / f"run_{name}.sh")],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(process)
        return process

    try:
        backend = launch("backend")
        connect_host = "127.0.0.1" if host == "0.0.0.0" else host
        url = f"http://{connect_host}:{port}"
        for _ in range(90):
            if backend.poll() is not None:
                raise RuntimeError("Backend exited. See studio/logs/backend.log.")
            try:
                with urlopen(url + "/api/health", timeout=1) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except (URLError, OSError, ValueError):
                time.sleep(0.5)
        else:
            raise RuntimeError("Backend did not become ready. See studio/logs/backend.log.")
        if dev:
            launch("frontend")
            url = f"http://{connect_host}:{os.environ.get('FRONTEND_PORT', '5173')}"
        print(f"Legoizer: {url}/", flush=True)
        print("Logs: studio/logs/. Press Ctrl+C to stop, or run bash studio/stop.sh.", flush=True)
        while True:
            for process in children:
                if process.poll() is not None:
                    raise RuntimeError("A service exited. See studio/logs/.")
            time.sleep(0.5)
    except StopRequested:
        return 0
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for process in children:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in children:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
        if STATE.is_file() and json.loads(STATE.read_text()) == state:
            STATE.unlink()
        lock.close()


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["run"]:
            sys.exit(run())
        if sys.argv[1:] == ["stop"]:
            sys.exit(stop())
        raise RuntimeError("Use services.py run|stop through the shell wrappers.")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Legoizer: {exc}", file=sys.stderr)
        sys.exit(1)
