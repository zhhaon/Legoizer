"""Registry tests: metadata must come from real source / YAML, not invention."""
from __future__ import annotations

import inspect

from studio.backend.registry.registry import (
    component_index,
    default_params,
    get_catalog,
    port_compatible,
)
from studio.backend.schemas.graph import GraphJson


def test_registry_loads_dynamically(catalog):
    assert len(catalog.components) >= 70
    assert {c.id for c in catalog.components} >= {
        "item.cnn", "item.cnncat", "user.ada", "user.gru", "predictor.dot",
        "predictor.deepfm", "dataset.mind", "training.matching_task",
    }
    assert all(component.id and component.name and component.description for component in catalog.components)
    assert len(catalog.categories) == 5
    assert not any(component.role == "evaluation" for component in catalog.components)
    assert {"GAUC", "NDCG", "F1"} <= {metric.name for metric in catalog.metrics}
    assert all(port["accepts"] for port in catalog.port_types)


def test_registry_discovers_real_python_classes(catalog):
    from loader.class_hub import ClassHub

    operators = ClassHub.operators()
    predictors = ClassHub.predictors()
    ids = {c.id for c in catalog.components}
    for key in ("cnn", "cnncat", "gru", "ada", "attention", "pooling"):
        assert key in operators.class_dict
        assert f"item.{key}" in ids or f"user.{key}" in ids
    for key in ("dot", "deepfm", "dcn", "din", "autoint", "finalmlp"):
        assert key in predictors.class_dict
        assert f"predictor.{key}" in ids


def test_cnn_parameters_match_real_config_class(catalog):
    cnn = component_index()["item.cnn"]
    names = {p.name for p in cnn.parameters}
    assert {"kernel_size", "dropout", "additive_hidden_size", "hidden_size"} <= names
    from model.operators.cnn_operator import CNNOperatorConfig

    signature = inspect.signature(CNNOperatorConfig.__init__)
    for name in ("kernel_size", "dropout", "additive_hidden_size"):
        assert name in signature.parameters
        param = next(p for p in cnn.parameters if p.name == name)
        assert param.default == signature.parameters[name].default


def test_meta_values_resolve_through_class_hub(catalog):
    from loader.class_hub import ClassHub

    operators = ClassHub.operators()
    predictors = ClassHub.predictors()
    for component in catalog.components:
        if not component.executable or component.backend_class is None:
            continue
        if component.role.startswith(("item", "user")):
            assert component.meta_value and component.meta_value.lower() in operators.class_dict
        elif component.role == "predictor":
            assert component.meta_value and component.meta_value.lower() in predictors.class_dict


def test_feature_annotations_are_not_in_library(catalog):
    assert not any(c.category == "feature" for c in catalog.components)
    assert not any(c["id"] == "feature" for c in catalog.categories)


def test_port_compatibility_is_semantic():
    assert port_compatible("ItemContent", "ItemContent")
    assert port_compatible("ItemEmbedding", "CandidateItemEmbedding")
    assert not port_compatible("Dataset", "ItemContent")
    assert not port_compatible("UserEmbedding", "ItemContent")


def test_default_params_are_json_serialisable():
    params = default_params("item.cnn")
    graph = GraphJson(nodes=[{"id": "n1", "component": "item.cnn", "params": params, "position": {"x": 0, "y": 0}}])
    assert graph.nodes[0].params["kernel_size"] == 3


def test_dataset_components_reference_existing_files(catalog, tmp_path):  # noqa: ARG001
    from studio.backend.config import REPO_ROOT

    for component in catalog.components:
        if component.category != "dataset":
            continue
        rel = next(p.default for p in component.parameters if p.name == "data_config")
        assert (REPO_ROOT / rel).is_file(), rel


def test_training_objectives_are_canonical(catalog):
    assert {c.id for c in catalog.components if c.role == "training"} == {
        "training.matching_task", "training.ranking_task",
    }


def test_legacy_training_nodes_preserve_parameters():
    from studio.backend.schemas.graph import GraphNode

    for old, canonical in (("bce_loss", "ranking_task"),
                           ("cross_entropy", "matching_task"),
                           ("negative_sampling", "matching_task")):
        node = GraphNode(id="objective", component=f"training.{old}", params={"neg_count": 7})
        assert node.component == f"training.{canonical}"
        assert node.params == {"neg_count": 7}
