"""Contract regressions for official Benchmark adapter; no live network required."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from studio.backend import benchmark as b


def record(signature="abc_123"):
    return {"signature": signature, "command": 'python trainer.py --data "config/recbench/mind.yaml" --model=config/model/naml.yaml --lm glove',
            "experiments": [
                {"seed": 1, "is_completed": True, "performance": {"GAUC": .6, "MRR": .4}},
                {"seed": 2, "is_completed": True, "performance": {"GAUC": .8, "bad": float("nan")}},
                {"seed": 3, "is_completed": False, "performance": {"GAUC": 99}},
                {"seed": 4, "is_completed": True, "performance": None},
            ]}


def test_statistics_and_command():
    result = b.summarize(record())
    assert (result["dataset"], result["model"], result["lm"]) == ("mind", "naml", "glove")
    assert result["completed"] == 3
    assert result["metrics"]["GAUC"] == pytest.approx({"mean": .7, "std": 2**.5 / 10, "count": 2})
    assert result["metrics"]["MRR"]["std"] is None
    assert "bad" not in result["metrics"]


def test_all_pages_cache_and_no_partial_results(monkeypatch):
    monkeypatch.setattr(b, "_cache", None)
    calls = []
    def fetch(path="", params=None):
        calls.append(params["page"])
        return {"page": params["page"], "total_page": 2, "total": 2,
                "evaluations": [record(str(params["page"]))]}
    monkeypatch.setattr(b, "_fetch", fetch)
    assert b.evaluations()["total"] == 2
    assert b.evaluations()["total"] == 2
    assert calls == [1, 2]
    monkeypatch.setattr(b, "_cache", None)
    monkeypatch.setattr(b, "_fetch", lambda **kw: {"evaluations": [], "total": 1, "total_page": 1})
    with pytest.raises(b.HTTPException) as exc:
        b.evaluations()
    assert exc.value.status_code == 502
    assert b._cache is None


def test_envelope_network_errors_and_detail(monkeypatch):
    monkeypatch.setattr(b.httpx, "get", lambda *a, **kw: b.httpx.Response(200, request=b.httpx.Request("GET", b.SOURCE), json={"code": 200, "body": {"signature": "abc", "configuration": {}}}))
    assert b._fetch("abc")["signature"] == "abc"
    app = FastAPI(); app.include_router(b.router)
    client = TestClient(app)
    assert client.get("/api/benchmark/evaluations/abc").json()["evaluation"]["configuration"] == {}
    assert client.get("/api/benchmark/evaluations/bad.dot").status_code == 422
    monkeypatch.setattr(b.httpx, "get", lambda *a, **kw: b.httpx.Response(200, request=b.httpx.Request("GET", b.SOURCE), json={"code": 400, "body": {}}))
    assert client.get("/api/benchmark/evaluations/abc").status_code == 502
    def timeout(*a, **kw):
        raise b.httpx.ReadTimeout("timeout")
    monkeypatch.setattr(b.httpx, "get", timeout)
    with pytest.raises(b.HTTPException):
        b._fetch()
