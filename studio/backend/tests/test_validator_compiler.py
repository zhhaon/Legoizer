"""Validator, compiler and Graph JSON round-trip tests."""
from __future__ import annotations

import copy

import pytest
import yaml

from studio.backend.compiler.compiler import compile_graph, import_yaml
from studio.backend.compiler.verify import verify_with_legommenders
from studio.backend.schemas.graph import GraphJson, GraphNode
from studio.backend.validator.validator import validate


def test_naml_template_is_valid(naml_graph):
    report = validate(GraphJson.model_validate(naml_graph))
    assert report.valid, [e.message for e in report.errors]
    assert report.summary.model_label == "NAML"
    assert report.summary.nodes == len(naml_graph["nodes"])


@pytest.mark.parametrize("template_id", ["nrms", "lstur", "deepfm", "dcn", "din", "miner"])
def test_all_shipped_templates_validate(templates, template_id):
    template = next(t for t in templates if t["id"] == template_id)
    assert template["available"], template.get("error")
    report = validate(GraphJson.model_validate(template["graph"]))
    assert report.valid, [e.message for e in report.errors]


def test_unknown_component_rejected(naml_graph):
    graph = copy.deepcopy(naml_graph)
    graph["nodes"][0]["component"] = "does.not.exist"
    report = validate(GraphJson.model_validate(graph))
    assert not report.valid
    assert any(e.code == "unknown_component" for e in report.errors)


def test_port_type_mismatch_rejected(naml_graph):
    graph = copy.deepcopy(naml_graph)
    # Wire a Dataset output into an ItemContent input.
    dataset = next(n for n in graph["nodes"] if n["component"].startswith("dataset."))
    item = next(n for n in graph["nodes"] if n["component"].startswith("item."))
    graph["edges"] = [
        e for e in graph["edges"]
        if not (e["source"] == dataset["id"] and e["target"] == item["id"] and e["targetHandle"] == "content")
    ]
    graph["edges"].append({
        "id": "bad_edge", "source": dataset["id"], "sourceHandle": "dataset",
        "target": item["id"], "targetHandle": "content",
    })
    report = validate(GraphJson.model_validate(graph))
    assert not report.valid
    assert any(e.code in ("port_type_mismatch", "missing_input") for e in report.errors)


def test_orphan_node_rejected(naml_graph):
    graph = copy.deepcopy(naml_graph)
    graph["nodes"].append({
        "id": "lonely", "component": "training.negative_sampling", "params": {},
        "position": {"x": 100, "y": 100},
    })
    report = validate(GraphJson.model_validate(graph))
    assert any(e.code == "orphan_node" and e.node_id == "lonely" for e in report.errors)


def test_cycle_rejected(naml_graph):
    graph = copy.deepcopy(naml_graph)
    a, b = graph["nodes"][0]["id"], graph["nodes"][1]["id"]
    graph["edges"].append({"id": "cycle_1", "source": b, "sourceHandle": "embedding", "target": a, "targetHandle": "dataset"})
    graph["edges"].append({"id": "cycle_2", "source": a, "sourceHandle": "dataset", "target": b, "targetHandle": "content"})
    report = validate(GraphJson.model_validate(graph))
    assert any(e.code == "cycle" for e in report.errors)


def test_parameter_type_and_range_checked(naml_graph):
    graph = copy.deepcopy(naml_graph)
    cnn = next(n for n in graph["nodes"] if n["component"] == "item.cnn")
    cnn["params"]["kernel_size"] = "not-an-integer"
    report = validate(GraphJson.model_validate(graph))
    assert any(e.code == "parameter_type" for e in report.errors)
    graph = copy.deepcopy(naml_graph)
    cnn = next(n for n in graph["nodes"] if n["component"] == "item.cnn")
    cnn["params"]["kernel_size"] = 0
    report = validate(GraphJson.model_validate(graph))
    assert any(e.code == "parameter_range" for e in report.errors)
    cnn["params"]["kernel_size"] = 33
    assert validate(GraphJson.model_validate(graph)).valid


def test_dimension_mismatch_detected(naml_graph):
    graph = copy.deepcopy(naml_graph)
    cnn = next(n for n in graph["nodes"] if n["component"] == "item.cnn")
    cnn["params"]["hidden_size"] = 128
    report = validate(GraphJson.model_validate(graph))
    assert any(e.code == "dimension_mismatch" for e in report.errors)
    assert any("Please" in e.fix or "Set" in e.fix for e in report.errors)


def test_runtime_dimension_checks_are_explicit(naml_graph):
    report = validate(GraphJson.model_validate(naml_graph))
    assert report.runtime_checks
    assert all(issue.severity == "info" for issue in report.runtime_checks)


def test_task_compatibility_uses_real_flags(naml_graph):
    graph = copy.deepcopy(naml_graph)
    # DIN has allow_matching=False in model/predictors/din_predictor.py.
    predictor = next(n for n in graph["nodes"] if n["component"] == "predictor.dot")
    predictor["component"] = "predictor.din"
    predictor["params"] = {"hidden_size": 256}
    report = validate(GraphJson.model_validate(graph))
    assert any("does not support the matching task" in e.message for e in report.errors)


def test_compile_produces_loadable_yaml(naml_graph):
    graph = GraphJson.model_validate(naml_graph)
    result = compile_graph(graph)
    parsed = yaml.safe_load(result.yaml_text)
    assert parsed["name"] == "NAML"
    assert parsed["meta"] == {"user": "Ada", "predictor": "Dot", "item": "CNN"}
    assert parsed["config"]["use_neg_sampling"] is True
    assert parsed["config"]["hidden_size"] == 256
    assert result.compatible_with_reference
    assert result.generated_file and (graph.graph_id in result.generated_file)


def test_compiled_config_has_no_invented_keys(naml_graph):
    import inspect

    from model.lego_config import LegoConfig

    result = compile_graph(GraphJson.model_validate(naml_graph))
    allowed = set(inspect.signature(LegoConfig.__init__).parameters) - {"self", "kwargs"}
    assert set(result.model_dict["config"]).issubset(allowed)


def test_compiled_yaml_verified_by_original_loader(naml_graph):
    result = compile_graph(GraphJson.model_validate(naml_graph))
    report = verify_with_legommenders(
        model_path=result.generated_file,
        data_config=result.data_config,
        exp_config=result.exp_config,
        embed_config=result.embed_config,
        hidden_size=256, batch_size=64, lr=0.001,
    )
    assert report["config_loaded"], report["errors"]
    assert report["classes_resolved"], report["errors"]
    assert report["configs_instantiated"], report["errors"]


def test_yaml_import_round_trip(naml_graph):
    result = compile_graph(GraphJson.model_validate(naml_graph))
    graph = import_yaml(result.yaml_text)
    assert graph.task == "matching"
    components = {n.component for n in graph.nodes}
    assert {"item.cnn", "user.ada", "predictor.dot"} <= components
    assert validate(graph).valid


def test_original_template_files_are_unchanged(naml_graph):
    from studio.backend.config import REPO_ROOT

    before = (REPO_ROOT / "config/model/naml.yaml").read_text(encoding="utf-8")
    compile_graph(GraphJson.model_validate(naml_graph))
    after = (REPO_ROOT / "config/model/naml.yaml").read_text(encoding="utf-8")
    assert before == after


def test_model_morph_naml_to_lstur(naml_graph):
    """End-to-end acceptance scenario: replace Additive Attention with GRU."""
    graph = copy.deepcopy(naml_graph)
    user_node = next(n for n in graph["nodes"] if n["component"] == "user.ada")
    user_node["component"] = "user.gru"
    user_node["params"] = {"hidden_size": 256, "num_layers": 1,
                           "inputer_config.use_cls_token": False, "inputer_config.use_sep_token": False}
    validated = validate(GraphJson.model_validate(graph))
    assert validated.valid, [e.message for e in validated.errors]
    assert validated.summary.model_label == "LSTUR"
    result = compile_graph(GraphJson.model_validate(graph))
    parsed = yaml.safe_load(result.yaml_text)
    assert parsed["meta"]["user"] == "GRU"
    assert parsed["name"] == "LSTUR"
    assert result.compatible_with_reference


def test_custom_combination_not_claimed_as_paper_model(naml_graph):
    graph = copy.deepcopy(naml_graph)
    predictor = next(n for n in graph["nodes"] if n["component"] == "predictor.dot")
    predictor["component"] = "predictor.dcn"
    predictor["params"] = {"hidden_size": 256, "cross_num": 3}
    report = validate(GraphJson.model_validate(graph))
    assert report.summary.model_label == "Custom Model"
    assert any(i.code == "model_identity" for i in report.warnings)


def test_legacy_metric_blocks_migrate_to_training(naml_graph):
    from studio.backend.compiler.compiler import build_exp_config

    graph = copy.deepcopy(naml_graph)
    graph["nodes"].append({"id": "old_metric", "component": "evaluation.ndcg", "params": {"k": 20}})
    graph["edges"].append({"id": "old_edge", "source": "predictor_1", "sourceHandle": "score",
                           "target": "old_metric", "targetHandle": "score"})
    migrated = GraphJson.model_validate(graph)
    assert all(node.id != "old_metric" for node in migrated.nodes)
    assert all(edge.target != "old_metric" for edge in migrated.edges)
    assert build_exp_config(migrated)["metrics"] == ["GAUC", "NDCG@20"]


def test_attention_head_count_is_validated_against_model_width(naml_graph):
    graph = copy.deepcopy(naml_graph)
    item = next(node for node in graph["nodes"] if node["component"] == "item.cnn")
    item["component"] = "item.attention"
    item["params"] = {"hidden_size": 256, "num_attention_heads": 3}
    report = validate(GraphJson.model_validate(graph))
    assert any(issue.node_id == item["id"] and "divisible" in issue.message for issue in report.errors)


def test_ranking_with_content_encoder_compiles(naml_graph):
    graph = GraphJson.model_validate(copy.deepcopy(naml_graph))
    graph.task = "ranking"
    objective = next(n for n in graph.nodes if n.component == "training.matching_task")
    objective.component = "training.ranking_task"
    objective.params.pop("neg_count", None)
    report = validate(graph)
    assert report.valid, [e.message for e in report.errors]
    config = compile_graph(graph, write_files=False).model_dict
    assert config["config"]["use_item_content"] is True
    assert config["config"]["use_neg_sampling"] is False
    assert config["meta"]["item"] == "CNN"


def test_matching_with_id_embeddings_compiles(templates):
    from studio.backend.registry.registry import default_params

    graph = GraphJson.model_validate(copy.deepcopy(next(t for t in templates if t["id"] == "deepfm")["graph"]))
    graph.task = "matching"
    predictor = next(n for n in graph.nodes if n.component.startswith("predictor."))
    predictor.component = "predictor.dot"
    predictor.params = default_params("predictor.dot")
    objective = next(n for n in graph.nodes if n.component.startswith("training."))
    objective.component = "training.matching_task"
    objective.params = default_params(objective.component)
    report = validate(graph)
    assert report.valid, [e.message for e in report.errors]
    config = compile_graph(graph, write_files=False).model_dict
    assert config["config"]["use_item_content"] is False
    assert config["config"]["use_neg_sampling"] is True
    assert "item" not in config["meta"]


def test_candidate_warning_does_not_block_compilation(naml_graph):
    graph = GraphJson.model_validate(copy.deepcopy(naml_graph))
    graph.edges = [e for e in graph.edges if e.targetHandle != "candidate"]
    report = validate(graph)
    assert report.valid, [e.message for e in report.errors]
    assert all(e.severity == "error" for e in report.errors)
    assert any(w.code == "missing_input" and w.severity == "warning" for w in report.warnings)
    compile_graph(graph, write_files=False)


def test_din_text_wiring_compiles(templates):
    from studio.backend.registry.registry import default_params
    from studio.backend.schemas.graph import GraphEdge

    graph = GraphJson.model_validate(copy.deepcopy(next(t for t in templates if t["id"] == "din")["graph"]))
    dataset = next(n for n in graph.nodes if n.component.startswith("dataset."))
    user = next(n for n in graph.nodes if n.component.startswith("user."))
    predictor = next(n for n in graph.nodes if n.component.startswith("predictor."))
    graph.nodes.append(GraphNode(id="item_text", component="item.pooling", params=default_params("item.pooling")))
    graph.edges = [e for e in graph.edges if not (
        (e.target == user.id and e.targetHandle == "history") or
        (e.target == predictor.id and e.targetHandle == "features")
    )]
    graph.edges.extend([
        GraphEdge(id="text_content", source=dataset.id, sourceHandle="content", target="item_text", targetHandle="content"),
        GraphEdge(id="text_history", source="item_text", sourceHandle="history", target=user.id, targetHandle="history"),
        GraphEdge(id="text_candidate", source="item_text", sourceHandle="candidate", target=predictor.id, targetHandle="candidate"),
    ])
    report = validate(graph)
    assert report.valid, [e.message for e in report.errors]
    config = compile_graph(graph, write_files=False).model_dict
    assert config["meta"] == {"user": "NullConcat", "predictor": "DIN", "item": "Pooling"}
    assert config["config"]["use_item_content"] is True
    assert config["config"]["use_neg_sampling"] is False


def test_legacy_feature_annotations_do_not_change_training(naml_graph):
    from studio.backend.compiler.compiler import build_model_config, build_exp_config

    baseline = GraphJson.model_validate(copy.deepcopy(naml_graph))
    legacy = copy.deepcopy(naml_graph)
    dataset_id = next(n["id"] for n in legacy["nodes"] if n["component"].startswith("dataset."))
    for key in ("user_id", "item_id", "item_text", "category", "user_history", "item_embedding", "user_embedding"):
        legacy["nodes"].append({"id": key, "component": f"feature.{key}"})
        legacy["edges"].append({"id": f"feature_edge_{key}", "source": dataset_id,
                                "sourceHandle": "dataset", "target": key, "targetHandle": "dataset"})
    migrated = GraphJson.model_validate(legacy)
    assert migrated.nodes == baseline.nodes
    assert migrated.edges == baseline.edges
    assert validate(migrated).valid
    assert build_model_config(migrated) == build_model_config(baseline)
    assert build_exp_config(migrated) == build_exp_config(baseline)
