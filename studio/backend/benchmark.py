"""Read-only adapter for the official lego-backend public API.

Contract: evaluation/views.py, evaluation/models.py and backend/urls.py in
https://github.com/legommenders/lego-backend (inspected 2026-11-17).
"""
from __future__ import annotations

import math
import shlex
import statistics
import threading
import time
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Path

SOURCE = "https://lego.qijiong.work/evaluations/"
TTL = 300
router = APIRouter(prefix="/api/benchmark", tags=["Official Benchmark"])
_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cached_at = 0.0


def _fetch(path: str = "", params: dict | None = None) -> dict:
    try:
        response = httpx.get(SOURCE + path, params=params, timeout=30, follow_redirects=True)
        response.raise_for_status()
        envelope = response.json()
        if not isinstance(envelope, dict) or envelope.get("code") != 200 or not isinstance(envelope.get("body"), dict):
            raise ValueError("Invalid Benchmark response")
        return envelope["body"]
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(502, "Official Benchmark is temporarily unavailable. Please retry.") from exc


def _params(command: str) -> dict[str, str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}
    params = {}
    for index, token in enumerate(tokens):
        if token.startswith("--"):
            key, separator, value = token[2:].partition("=")
            if separator:
                params[key] = value
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                params[key] = tokens[index + 1]
    return params


def summarize(evaluation: dict) -> dict:
    params = _params(evaluation.get("command", ""))
    values: dict[str, list[float]] = {}
    completed = 0
    for experiment in evaluation.get("experiments", []):
        if not experiment.get("is_completed"):
            continue
        completed += 1
        for name, value in (experiment.get("performance") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.setdefault(name, []).append(value)
    def stem(key):
        return params.get(key, "").replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return {
        **evaluation,
        "model": stem("model"), "dataset": stem("data"), "lm": params.get("lm", ""),
        "params": params, "completed": completed,
        "metrics": {name: {"mean": statistics.mean(items),
                            "std": statistics.stdev(items) if len(items) > 1 else None,
                            "count": len(items)} for name, items in values.items()},
    }


@router.get("/evaluations")
def evaluations():
    global _cache, _cached_at
    with _lock:
        if _cache is not None and time.monotonic() - _cached_at < TTL:
            return _cache
        first = _fetch(params={"page": 1, "page_size": 100})
        try:
            pages = int(first["total_page"])
            if not 1 <= pages <= 1000 or not isinstance(first["evaluations"], list):
                raise ValueError("Invalid pagination")
            items = list(first["evaluations"])
            for page in range(2, pages + 1):
                result = _fetch(params={"page": page, "page_size": 100})
                if result.get("page") != page or result.get("total") != first["total"]:
                    raise ValueError("Benchmark changed during pagination; retry")
                items.extend(result["evaluations"])
            if len(items) != first["total"] or len({item["signature"] for item in items}) != len(items):
                raise ValueError("Incomplete Benchmark response")
            result = {"evaluations": [summarize(item) for item in items], "total": len(items),
                      "source": SOURCE, "fetched_at": time.time(), "cache_seconds": TTL}
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(502, "Incomplete Benchmark response. Please retry.") from exc
        _cache, _cached_at = result, time.monotonic()
        return result


@router.get("/evaluations/{signature}")
def evaluation_detail(signature: str = Path(pattern=r"^[A-Za-z0-9_-]{1,128}$")):
    return {"evaluation": _fetch(quote(signature, safe="")), "source": SOURCE + quote(signature, safe="")}
