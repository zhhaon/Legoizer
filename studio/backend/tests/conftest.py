"""Shared pytest fixtures for the Studio backend tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def catalog():
    from studio.backend.registry.registry import get_catalog

    return get_catalog()


@pytest.fixture(scope="session")
def templates():
    from studio.backend.templates.templates import get_templates

    return get_templates()


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from studio.backend.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def naml_graph(templates) -> dict:
    template = next(t for t in templates if t["id"] == "naml")
    assert template["available"], template.get("error")
    return template["graph"]
