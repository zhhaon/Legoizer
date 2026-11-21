"""End-to-end integration tests against the *original* Legommenders code.

The acceptance workflow is exercised without a GUI:
    NAML template -> validate -> compile YAML -> original config loader ->
    ClassHub resolution -> real config_class instantiation -> morph to LSTUR.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from studio.backend.compiler.compiler import compile_graph
from studio.backend.compiler.verify import verify_with_legommenders
from studio.backend.schemas.graph import GraphJson
from studio.backend.validator.validator import validate


def test_end_to_end_naml_to_lstur(naml_graph):
    """The most important acceptance path, as a reproducible backend test."""
    graph = GraphJson.model_validate(naml_graph)

    # 1. Validate
    report = validate(graph)
    assert report.valid, [e.message for e in report.errors]
    assert report.summary.model_label == "NAML"

    # 2. Compile
    result = compile_graph(graph)
    assert result.compatible_with_reference
    parsed = yaml.safe_load(result.yaml_text)
    assert parsed["meta"] == {"user": "Ada", "predictor": "Dot", "item": "CNN"}

    # 3. Original config loading mechanism
    verification = verify_with_legommenders(
        model_path=result.generated_file,
        data_config=result.data_config,
        exp_config=result.exp_config,
        embed_config=result.embed_config,
        hidden_size=graph.train.hidden_size,
        batch_size=graph.train.batch_size,
        lr=graph.train.lr,
    )
    assert verification["config_loaded"], verification["errors"]
    assert verification["classes_resolved"], verification["errors"]

    # 4. Model components can be instantiated from the compiled config
    assert verification["configs_instantiated"], verification["errors"]
    assert any("LegoConfig instantiated" in check for check in verification["checks"])
    assert any("CNNOperator.config_class" in check for check in verification["checks"])
    assert any("AdaOperator.config_class" in check for check in verification["checks"])
    assert any("DotPredictor.config_class" in check for check in verification["checks"])

    # 5. Model Morphing: replace Additive Attention with GRU
    morphed = copy.deepcopy(naml_graph)
    user_node = next(n for n in morphed["nodes"] if n["component"] == "user.ada")
    user_node["component"] = "user.gru"
    user_node["params"] = {
        "hidden_size": 256, "num_layers": 1,
        "inputer_config.use_cls_token": False, "inputer_config.use_sep_token": False,
    }
    morphed_graph = GraphJson.model_validate(morphed)
    morphed_report = validate(morphed_graph)
    assert morphed_report.valid, [e.message for e in morphed_report.errors]
    assert morphed_report.summary.model_label == "LSTUR"
    morphed_result = compile_graph(morphed_graph)
    assert yaml.safe_load(morphed_result.yaml_text)["meta"]["user"] == "GRU"
    assert morphed_result.compatible_with_reference
    morphed_verification = verify_with_legommenders(
        model_path=morphed_result.generated_file,
        data_config=morphed_result.data_config,
        exp_config=morphed_result.exp_config,
        embed_config=morphed_result.embed_config,
        hidden_size=256, batch_size=64, lr=0.001,
    )
    assert morphed_verification["configs_instantiated"], morphed_verification["errors"]
    assert any("GRUOperator.config_class" in check for check in morphed_verification["checks"])


def test_cli_config_loading_still_works():
    """The Studio must not break the original CLI configuration mechanism."""
    from trainer import get_configurations

    cfg = get_configurations(dict(
        data="config/data/mind.yaml",
        model="config/model/naml.yaml",
        embed="config/embed/glove.yaml",
        exp="config/exp/default.yaml",
        hidden_size=256,
        item_hidden_size="${hidden_size}$",
        batch_size=64,
        lr=0.001,
    ))
    assert cfg.model.meta.item == "CNN"
    assert cfg.model.config.hidden_size == 256
    assert cfg.data.name == "mind"


def test_training_environment_is_recorded(client):
    import time

    from studio.backend.executor import executor
    from studio.backend.schemas.graph import GraphJson as GJ
    from studio.backend.templates.templates import get_template
    from utils.config_init import DataInit

    raw = DataInit.get("mind", default="")
    if not (raw and (Path(raw) / "train" / "behaviors.tsv").is_file()):
        import pytest
        pytest.skip("Real MIND raw data is not configured on this machine")

    graph = GJ.model_validate(get_template("naml-demo")["graph"])
    graph.train.accumulate_batch = 3
    started = executor.start_training(graph)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and started.status not in ("completed", "failed", "cancelled"):
        time.sleep(0.5)
    if started.status in ("pending", "running"):
        executor.cancel(started.run_id)
        raise AssertionError("Quick demo did not finish in 180 seconds")
    assert started.status == "completed", started.error
    assert set(started.metrics) >= {"GAUC", "MRR", "NDCG@5", "NDCG@10"}
    assert started.params["accumulate_batch"] == 3
    assert started.result_file
