"""HTTP API tests using FastAPI's TestClient."""
from __future__ import annotations

from copy import deepcopy

import yaml


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_components_endpoint(client):
    body = client.get("/api/components").json()
    assert len(body["components"]) >= 70
    assert not any(c["id"].startswith("feature.") for c in body["components"])
    assert body["generated_from"].startswith("loader.class_hub.ClassHub")
    cnn = next(c for c in body["components"] if c["id"] == "item.cnn")
    assert cnn["backend_class"].endswith("CNNOperator")
    assert any(p["name"] == "kernel_size" for p in cnn["parameters"])


def test_templates_endpoint(client):
    body = client.get("/api/templates").json()
    ids = {t["id"] for t in body["templates"]}
    assert {"naml", "nrms", "lstur", "deepfm", "dcn", "din"} <= ids
    assert all(t["available"] for t in body["templates"])
    naml = next(t for t in body["templates"] if t["id"] == "naml")
    assert naml["source_yaml"] == "config/model/naml.yaml"
    assert naml["resolved_config"]["meta"]["item"] == "CNN"


def test_validate_endpoint(client, naml_graph):
    response = client.post("/api/graph/validate", json=naml_graph)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["summary"]["model_label"] == "NAML"


def test_dataset_inputs_preview_and_validation(client, naml_graph):
    naml_graph = deepcopy(naml_graph)
    dataset = next(node for node in naml_graph["nodes"] if node["component"].startswith("dataset."))
    preview = client.post("/api/graph/dataset-inputs", json=naml_graph)
    assert preview.status_code == 200
    assert [row["column"] for row in preview.json()["defaults"]] == ["title@glove", "category"]

    dataset["params"]["item_inputs"] = [
        {"column": "category", "truncate": None},
        {"column": "title@glove", "truncate": 18},
    ]
    assert client.post("/api/graph/validate", json=naml_graph).json()["valid"]
    dataset["params"]["item_inputs"][1]["column"] = "missing_item_feature"
    report = client.post("/api/graph/validate", json=naml_graph).json()
    assert any(error["code"] == "dataset_inputs" and error["node_id"] == dataset["id"] for error in report["errors"])


def test_dataset_external_hf_resources_preserve_the_item_encoder(client, naml_graph):
    graph = deepcopy(naml_graph)
    dataset = next(node for node in graph["nodes"] if node["component"].startswith("dataset."))
    dataset["params"].update(token_source="external_hf", external_model_id="example/token-weights")
    preview = client.post("/api/graph/dataset-inputs", json=graph).json()
    assert [entry["column"] for entry in preview["defaults"]] == ["title@external_hf", "category"]
    result = client.post("/api/graph/compile", json=graph)
    assert result.status_code == 200, result.text
    assert result.json()["model_dict"]["meta"]["item"] == "CNN"
    assert result.json()["embed_config"] == "config/embed/null.yaml"

    dataset["params"]["external_model_id"] = ""
    errors = client.post("/api/graph/validate", json=graph).json()["errors"]
    assert any(entry["code"] == "external_hf_source" and entry["node_id"] == dataset["id"] for entry in errors)

    item = next(node for node in graph["nodes"] if node["component"].startswith("item."))
    item.update(component="item.externalhf", params={
        "model_id": "example/encoder", "hidden_size": graph["train"]["hidden_size"],
    })
    preview = client.post("/api/graph/dataset-inputs", json=graph).json()
    assert [entry["column"] for entry in preview["defaults"]] == ["title@external_hf"]
    assert client.post("/api/graph/validate", json=graph).json()["valid"]


def test_external_hf_compile_and_dataset_inputs(client, naml_graph):
    graph = deepcopy(naml_graph)
    item = next(node for node in graph["nodes"] if node["component"].startswith("item."))
    item.update(component="item.externalhf", params={
        "model_id": "example/text-model", "encoding_mode": "token_embeddings",
        "hidden_size": graph["train"]["hidden_size"],
    })
    preview = client.post("/api/graph/dataset-inputs", json=graph)
    assert preview.status_code == 200
    assert preview.json()["defaults"] == [{"column": "title@external_hf", "truncate": 30}]
    response = client.post("/api/graph/compile", json=graph)
    assert response.status_code == 200, response.text
    generated = yaml.safe_load(response.json()["yaml_text"])
    assert generated["meta"]["item"] == "ExternalHF"
    assert generated["config"]["item_config"]["model_id"] == "example/text-model"
    assert "checkpoint_dir" not in generated["config"]["item_config"]
    assert "inputer_config" not in generated["config"]["item_config"]

    item["params"]["model_id"] = "/etc"
    report = client.post("/api/graph/validate", json=graph).json()
    assert any(error["code"] == "external_hf_source" for error in report["errors"])


def test_compile_endpoint(client, naml_graph):
    response = client.post("/api/graph/compile", json=naml_graph)
    assert response.status_code == 200
    body = response.json()
    assert body["model_label"] == "NAML"
    assert body["compatible_with_reference"] is True
    parsed = yaml.safe_load(body["yaml_text"])
    assert parsed["meta"]["predictor"] == "Dot"
    assert body["generated_file"].startswith("studio/backend/workspace/generated/")


def test_invalid_graph_is_rejected_by_compile(client, naml_graph):
    graph = {**naml_graph, "nodes": []}
    response = client.post("/api/graph/validate", json=graph)
    assert response.json()["valid"] is False
    response = client.post("/api/graph/compile", json=graph)
    assert response.status_code == 422


def test_import_yaml_endpoint(client, naml_graph):
    compiled = client.post("/api/graph/compile", json=naml_graph).json()
    response = client.post("/api/graph/import", json={"yaml": compiled["yaml_text"]})
    assert response.status_code == 200
    body = response.json()
    assert body["validation"]["valid"] is True
    assert any(n["component"] == "item.cnn" for n in body["graph"]["nodes"])


def test_import_graph_json_endpoint(client, naml_graph):
    response = client.post("/api/graph/import", json={"graph": naml_graph})
    assert response.status_code == 200
    assert response.json()["graph"]["name"] == "NAML"


def test_compatible_endpoint(client):
    response = client.get("/api/components/user.ada/compatible?direction=upstream")
    assert response.status_code == 200
    ids = {r["id"] for r in response.json()["results"]}
    assert "dataset.mind" in ids or "item.cnn" in ids
    assert client.get("/api/components/nope/compatible").status_code == 404


def test_environment_endpoint(client):
    body = client.get("/api/environment").json()
    assert body["python"]
    assert body["torch"]
    assert isinstance(body["cuda_available"], bool)
    assert body["trainer_script"] is True
    assert isinstance(body["warnings"], list)
    assert any(row["config"] == "config/data/mind.yaml" for row in body["datasets"])


def test_diagnostics_verify_endpoint(client, naml_graph):
    response = client.post("/api/diagnostics/verify", json=naml_graph)
    assert response.status_code == 200
    assert response.json()["compatible"] is True


def test_save_and_load_graph(client, naml_graph):
    graph = {**naml_graph, "graph_id": "test_saved_graph", "name": "Saved NAML"}
    assert client.post("/api/graphs/save", json={"graph": graph}).json()["saved"] is True
    listing = client.get("/api/graphs").json()["graphs"]
    assert any(g["graph_id"] == "test_saved_graph" for g in listing)
    loaded = client.get("/api/graphs/test_saved_graph").json()
    assert loaded["graph"]["name"] == "Saved NAML"
    assert loaded["validation"]["valid"] is True


def test_reference_yaml_is_whitelisted(client):
    assert client.get("/api/reference/config/model/naml.yaml").status_code == 200
    # Encoded traversal stays inside the /api/reference route and must be rejected.
    assert client.get("/api/reference/config/model/%2e%2e%2f%2e%2e%2fREADME.md").status_code == 403
    assert client.get("/api/reference/config/data/mind.yaml").status_code == 403


def test_train_preflight_reports_real_blockers(client, naml_graph, monkeypatch):
    from studio.backend.executor import executor

    monkeypatch.setattr(executor, "tables_available", lambda *_: False)
    monkeypatch.setattr(executor, "table_ready", lambda _: False)
    response = client.post("/api/train", json={"graph": naml_graph, "preflight_only": True})
    assert response.status_code == 200
    body = response.json()
    assert body["blockers"], "expected real environment blockers"
    assert body["status"] == "failed"
    assert body["metrics"] == {}
    assert body["command"] == []
    assert any("Dataset files are missing" in b or "embedding" in b for b in body["blockers"])


def test_train_rejects_invalid_graph(client, naml_graph):
    graph = {**naml_graph, "nodes": []}
    response = client.post("/api/train", json={"graph": graph})
    assert response.status_code == 422


def test_runs_endpoint(client):
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert isinstance(response.json()["runs"], list)


def test_hf_token_requires_secure_transport_and_is_not_in_the_run(client, naml_graph, monkeypatch):
    from studio.backend.executor import executor
    from studio.backend.schemas.runs import RunRecord

    token = "hf_studioTest123456789"
    received = []

    def start(graph, *, preflight_only, hf_token):
        received.append(hf_token)
        assert hf_token not in graph.model_dump_json()
        return RunRecord(run_id="token-test")

    monkeypatch.setattr(executor, "start_training", start)
    headers = {"X-Hugging-Face-Token": token}
    body = {"graph": naml_graph}
    rejected = client.post("http://studio.example/api/train", json=body, headers=headers)
    assert rejected.status_code == 403
    assert not received
    accepted = client.post("https://studio.example/api/train", json=body, headers=headers)
    assert accepted.status_code == 200
    assert received == [token]
    assert token not in accepted.text
