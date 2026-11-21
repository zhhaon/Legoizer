"""A stop request must never signal an unrelated or reused process ID."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def services(tmp_path, monkeypatch):
    # run() changes cwd; register restoration even when its preflight raises.
    monkeypatch.chdir(Path.cwd())
    path = Path(__file__).resolve().parents[2] / "scripts/services.py"
    spec = importlib.util.spec_from_file_location("legoizer_test_services", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "STATE", tmp_path / "services.json")
    return module


def test_no_manager_is_a_noop(services, monkeypatch):
    monkeypatch.setattr(services.os, "kill", lambda *_: pytest.fail("No manager exists"))
    assert services.stop() == 0


def test_reused_pid_is_not_signalled(services, monkeypatch):
    services.STATE.write_text('{"pid": 12345, "starttime": "old"}')
    monkeypatch.setattr(services, "identity", lambda _: "new")
    monkeypatch.setattr(services.os, "kill", lambda *_: pytest.fail("PID was reused"))
    assert services.stop() == 0


def test_missing_frontend_fails_before_launching_anything(services, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "ROOT", tmp_path)
    monkeypatch.setattr(services, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setenv("STUDIO_DEV", "0")
    monkeypatch.setattr(services.subprocess, "Popen", lambda *_a, **_k: pytest.fail("No build exists"))
    with pytest.raises(RuntimeError, match="Frontend is not built"):
        services.run()
