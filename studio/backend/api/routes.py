"""HTTP API for Legoizer."""
from __future__ import annotations
from studio.backend.i18n import localized


import asyncio
import json
import re
import time
from typing import Any

import yaml
from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from studio.backend.compiler.compiler import compile_graph, import_yaml
from studio.backend.config import GRAPHS_DIR, REPO_ROOT
from studio.backend.dataset_inputs import resolve_dataset_inputs
from studio.backend.executor import executor
from studio.backend.registry.registry import get_catalog, get_component
from studio.backend.schemas.graph import GraphJson
from studio.backend.schemas.runs import EvaluateRequest, RunRecord, StartRunRequest
from studio.backend.templates.templates import get_template, get_templates
from studio.backend.validator.validator import compatible_components_for, validate

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "Legoizer", "time": time.time()}


@router.get("/components")
def components() -> dict[str, Any]:
    return get_catalog().model_dump(mode="json")


@router.get("/components/{component_id}/compatible")
def compatible(component_id: str, direction: str = Query(default="downstream", pattern="^(upstream|downstream)$")):
    if get_component(component_id) is None:
        raise HTTPException(status_code=404, detail=localized('backend.5ccdbe8d865a', "Unknown component", {}))
    return {"component": component_id, "direction": direction,
            "results": compatible_components_for(component_id, direction)}


@router.get("/templates")
def templates() -> dict[str, Any]:
    items = []
    for template in get_templates():
        public = {k: v for k, v in template.items() if k not in ("load_kwargs",)}
        items.append(public)
    return {"templates": items}


@router.get("/templates/{template_id}")
def template_detail(template_id: str):
    item = get_template(template_id)
    if not item:
        raise HTTPException(status_code=404, detail=localized('backend.43f6602a29c0', "Unknown template", {}))
    return item


@router.post("/graph/validate")
def graph_validate(graph: GraphJson) -> dict[str, Any]:
    return validate(graph).model_dump(mode="json")


@router.post("/graph/dataset-inputs")
def graph_dataset_inputs(graph: GraphJson) -> dict[str, Any]:
    try:
        return resolve_dataset_inputs(graph, use_override=False)
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/graph/compile")
def graph_compile(graph: GraphJson) -> dict[str, Any]:
    try:
        return compile_graph(graph).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ImportPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph: dict[str, Any] | None = None
    yaml: str | None = None


@router.post("/graph/import")
def graph_import(payload: ImportPayload) -> dict[str, Any]:
    try:
        if payload.yaml:
            graph = import_yaml(payload.yaml)
        elif payload.graph:
            graph = GraphJson.model_validate(payload.graph)
        else:
            raise ValueError("Provide either `graph` (Graph JSON) or `yaml` (Legommenders model YAML).")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=localized('backend.507dfd8b961a', f"Import failed: {type(exc).__name__}: {exc}", {'p0': f'{type(exc).__name__}', 'p1': f'{exc}'})) from exc
    return {"graph": graph.model_dump(mode="json"), "validation": validate(graph).model_dump(mode="json")}


@router.get("/environment")
def environment() -> dict[str, Any]:
    return executor.environment_report()


@router.post("/diagnostics/verify")
def diagnostics_verify(graph: GraphJson) -> dict[str, Any]:
    """A real, GPU-free test: compile and load through the original framework."""
    try:
        result = compile_graph(graph)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "model_label": result.model_label,
        "yaml_text": result.yaml_text,
        "notes": result.notes,
        "note_translations": result.note_translations,
        "compatible": result.compatible_with_reference,
        "validation": result.validation.model_dump(mode="json"),
    }


@router.post("/train")
def train(payload: StartRunRequest, request: Request) -> dict[str, Any]:
    token = request.headers.get("X-Hugging-Face-Token")
    if token:
        loopback = {"127.0.0.1", "::1", "localhost"}
        if request.url.scheme != "https" and not (
            request.client and request.client.host in loopback and request.url.hostname in loopback
        ):
            raise HTTPException(status_code=403, detail=localized('backend.065e9883ffcd', "Use HTTPS or an SSH tunnel to submit a Hugging Face token.", {}))
        if not re.fullmatch(r"hf_[A-Za-z0-9]{8,256}", token):
            raise HTTPException(status_code=422, detail=localized('backend.fef8bdea41ae', "Enter a valid Hugging Face user access token.", {}))

    try:
        graph = GraphJson.model_validate(payload.graph)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=localized('backend.cd98eb7d821e', f"Invalid Graph JSON: {exc}", {'p0': f'{exc}'})) from exc
    report = validate(graph)
    if not report.valid:
        raise HTTPException(status_code=422, detail={"message": "Model graph is not valid",
                                                    "errors": [e.model_dump() for e in report.errors]})
    try:
        record = executor.start_training(graph, preflight_only=payload.preflight_only, hf_token=token)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@router.get("/train/{run_id}/status")
def train_status(run_id: str) -> dict[str, Any]:
    record = executor.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {}))
    return record.model_dump(mode="json")


@router.get("/train/{run_id}/logs")
def train_logs(run_id: str, after: int = 0) -> dict[str, Any]:
    try:
        return executor.logs_since(run_id, max(0, after))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {})) from exc


@router.post("/train/{run_id}/cancel")
def train_cancel(run_id: str) -> dict[str, Any]:
    try:
        return executor.cancel(run_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {})) from exc


@router.get("/train/{run_id}/stream")
async def train_stream(run_id: str):
    """Server-Sent Events: real-time log lines + status transitions."""
    from fastapi.responses import StreamingResponse

    if executor.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {}))

    async def generator():
        cursor = 0
        last_status = ""
        while True:
            record = executor.get_run(run_id)
            if record is None:
                break
            payload = executor.logs_since(run_id, cursor)
            for line in payload["lines"]:
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
            cursor = payload["next"]
            if record.status != last_status:
                last_status = record.status
                yield f"data: {json.dumps({'type': 'status', 'status': record.status, 'metrics': record.metrics, 'error': record.error})}\n\n"
            if record.status in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.6)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/evaluate")
def evaluate(payload: EvaluateRequest) -> dict[str, Any]:
    source: RunRecord | None = None
    if payload.run_id:
        source = executor.get_run(payload.run_id)
        if source is None:
            raise HTTPException(status_code=404, detail=localized('backend.e183936c6ddb', "Unknown source run", {}))
        if source.status != "completed" or not source.metrics:
            raise HTTPException(status_code=409, detail=localized('backend.d4f955ecad9a', "Only successfully completed runs can be evaluated.", {}))
    raw = payload.graph or (source.graph if source else None)
    if not raw:
        raise HTTPException(status_code=422, detail=localized('backend.b65b63d0f601', "Provide `graph` or a completed `run_id`.", {}))
    try:
        graph = GraphJson.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    report = validate(graph)
    if not report.valid:
        raise HTTPException(status_code=422, detail={"message": "Model graph is not valid",
                                                    "errors": [e.model_dump() for e in report.errors]})
    try:
        record = executor.start_evaluation(graph, source_run=source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record.model_dump(mode="json")


@router.get("/runs")
def runs() -> dict[str, Any]:
    return {"runs": [r.model_dump(mode="json") for r in executor.list_runs()]}


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    record = executor.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {}))
    return record.model_dump(mode="json")


@router.get("/runs/{run_id}/model.yaml", response_class=PlainTextResponse)
def run_model_yaml(run_id: str) -> str:
    record = executor.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {}))
    if record.model_yaml:
        return record.model_yaml
    graph = GraphJson.model_validate(record.graph) if record.graph else None
    return compile_graph(graph).yaml_text if graph else ""


@router.get("/runs/{run_id}/graph.json")
def run_graph_json(run_id: str) -> dict[str, Any]:
    record = executor.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail=localized('backend.9b3588e3c308', "Unknown run", {}))
    return record.graph or {}


# --------------------------------------------------------------------------- #
# Local project persistence (JSON files in the server-controlled workspace).
# --------------------------------------------------------------------------- #
class SaveGraphPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph: dict[str, Any]
    overwrite: bool = True


@router.post("/graphs/save")
def save_graph(payload: SaveGraphPayload) -> dict[str, Any]:
    try:
        graph = GraphJson.model_validate(payload.graph)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in graph.graph_id)[:80]
    path = GRAPHS_DIR / f"{safe}.json"
    if path.exists() and not payload.overwrite:
        raise HTTPException(status_code=409, detail=localized('backend.d77495d6e0a2', "Graph id already exists", {}))
    path.write_text(json.dumps(graph.model_dump(mode="json"), indent=2), encoding="utf-8")
    return {"saved": True, "graph_id": graph.graph_id, "path": str(path.relative_to(REPO_ROOT))}


@router.get("/graphs")
def list_graphs() -> dict[str, Any]:
    items = []
    for path in sorted(GRAPHS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append({"graph_id": data.get("graph_id"), "name": data.get("name"),
                          "task": data.get("task"), "nodes": len(data.get("nodes") or []),
                          "saved_at": path.stat().st_mtime})
        except Exception:
            continue
    return {"graphs": items}


@router.get("/graphs/{graph_id}")
def load_graph(graph_id: str) -> dict[str, Any]:
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in graph_id)[:80]
    path = GRAPHS_DIR / f"{safe}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=localized('backend.c5f4ad4cb5d0', "Saved graph not found", {}))
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = GraphJson.model_validate(data)
    return {"graph": graph.model_dump(mode="json"), "validation": validate(graph).model_dump(mode="json")}


@router.get("/reference/{relative_path:path}", response_class=PlainTextResponse)
def reference_yaml(relative_path: str) -> str:
    """Serve a *whitelisted* shipped configuration for YAML comparison."""
    allowed = (REPO_ROOT / "config/model").resolve()
    target = (REPO_ROOT / relative_path).resolve()
    if not target.is_relative_to(allowed) or not target.is_file() or target.suffix != ".yaml":
        raise HTTPException(status_code=403, detail=localized('backend.5b0bfda111c2', "Only shipped model configurations may be read", {}))
    return target.read_text(encoding="utf-8")


class ResourceRequest(BaseModel):
    component: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/resources/status")
def resource_status(items: list[ResourceRequest]):
    from studio.backend.resources import inspect
    try:
        return {"items": [inspect(item.component, item.params) for item in items]}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/resources/download")
def resource_download(item: ResourceRequest, request: Request):
    from studio.backend.resources import acquire
    try:
        return acquire(item.component, item.params, request.headers.get("X-Hugging-Face-Token"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/resources/delete")
def resource_delete(item: ResourceRequest):
    from studio.backend.resources import remove
    try:
        return remove(item.component, item.params)
    except (ValueError, OSError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    try:
        executor.delete_run(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown run") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": True, "run_id": run_id}
