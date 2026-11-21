"""Model templates derived from the repository's real YAML files.

Every template is produced by running the *original* ``trainer.get_configurations``
against the shipped YAML, then mapping the resolved ``model`` section back into
Graph JSON.  Therefore the parameters shown in the UI are the real ones.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from studio.backend.compiler.compiler import import_yaml
from studio.backend.config import REPO_ROOT
from studio.backend.schemas.graph import GraphEdge, GraphJson, GraphNode, Position, TrainConfig

TEMPLATE_DEFS: list[dict[str, Any]] = [
    {"id": "naml-demo", "name": "NAML Quick Demo", "yaml": "config/model/naml.yaml", "task": "matching",
     "description": "Build and train NAML on a fixed sample of real MIND-small users.",
     "paper": "NAML (IJCAI 2019)", "featured": False, "quick_demo": True},
    {"id": "naml", "name": "NAML", "yaml": "config/model/naml.yaml", "task": "matching", "year": 2019,
     "description": "CNN Item Encoder + Additive Attention User Encoder + Dot Predictor.",
     "paper": "NAML (IJCAI 2019), arXiv:1907.05576", "featured": True},
    {"id": "nrms", "name": "NRMS", "yaml": "config/model/nrms.yaml", "task": "matching", "year": 2019,
     "description": "Multi-head self-attention on both news and user history + Dot Predictor.",
     "paper": "NRMS (EMNLP 2019)", "featured": True},
    {"id": "lstur", "name": "LSTUR", "yaml": "config/model/lstur.yaml", "task": "matching", "year": 2019,
     "description": "CNNCat Item Encoder + GRU User Encoder + Dot Predictor.",
     "paper": "LSTUR (ACL 2019)", "featured": False},
    {"id": "deepfm", "name": "DeepFM", "yaml": "config/model/deepfm_id.yaml", "task": "ranking", "year": 2017,
     "description": "ID-based ranking: Pooling user operator + DeepFM predictor (no content encoder).",
     "paper": "DeepFM (IJCAI 2017)", "featured": True},
    {"id": "dcn", "name": "DCN", "yaml": "config/model/dcn_id.yaml", "task": "ranking", "year": 2017,
     "description": "ID-based ranking: Pooling + Deep & Cross Network.",
     "paper": "DCN (KDD 2017)", "featured": False},
    {"id": "din", "name": "DIN", "yaml": "config/model/din_id.yaml", "task": "ranking", "year": 2017,
     "description": "ID-based ranking: NullConcat user operator + Deep Interest Network.",
     "paper": "DIN (KDD 2018)", "featured": False},
    {"id": "pnn", "name": "PNN", "yaml": "config/model/pnn_id.yaml", "task": "ranking", "year": 2016,
     "description": "ID-based ranking: Pooling + Product-based Neural Network.", "paper": "PNN (ICDM 2016)", "featured": False},
    {"id": "autoint", "name": "AutoInt", "yaml": "config/model/autoint_id.yaml", "task": "ranking", "year": 2018,
     "description": "ID-based ranking: Pooling + AutoInt interacting layer.", "paper": "AutoInt (CIKM 2019)", "featured": False},
    {"id": "dcnv2", "name": "DCNv2", "yaml": "config/model/dcnv2_id.yaml", "task": "ranking", "year": 2020,
     "description": "ID-based ranking: Pooling + improved Deep & Cross Network.", "paper": "DCNv2 (WWW 2021)", "featured": False},
    {"id": "masknet", "name": "MaskNet", "yaml": "config/model/masknet_id.yaml", "task": "ranking", "year": 2021,
     "description": "ID-based ranking: Pooling + MaskNet.", "paper": "MaskNet (DLP-KDD 2021)", "featured": False},
    {"id": "gdcn", "name": "GDCN", "yaml": "config/model/gdcn_id.yaml", "task": "ranking", "year": 2023,
     "description": "ID-based ranking: Pooling + Gated Deep & Cross Network.", "paper": "GDCN (2023)", "featured": False},
    {"id": "finalmlp", "name": "FinalMLP", "yaml": "config/model/finalmlp_id.yaml", "task": "ranking", "year": 2023,
     "description": "ID-based ranking: Pooling + two-stream FinalMLP.", "paper": "FinalMLP (KDD 2023)", "featured": False},
    {"id": "miner", "name": "MINER", "yaml": "config/model/miner.yaml", "task": "matching", "year": 2022,
     "description": "Transformer Item Encoder + PolyAttention User Encoder + MINER predictor.",
     "paper": "MINER (Findings of ACL 2022)", "featured": False},
]

_EVALS_BY_TASK = {
    "matching": ["GAUC", "MRR", "NDCG@5", "NDCG@10"],
    "ranking": ["AUC", "LogLoss", "GAUC", "Recall@10"],
}


def _resolved_model(relative: str, task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from trainer import get_configurations  # noqa: WPS433

    data = "config/data/mind.yaml" if task == "matching" else "config/recbench/mind.yaml"
    kwargs: dict[str, Any] = dict(
        data=data,
        model=relative,
        embed="config/embed/glove.yaml" if task == "matching" else "config/embed/null.yaml",
        exp="config/exp/default.yaml",
        hidden_size=256,
        item_hidden_size="${hidden_size}$",
        batch_size=64,
        lr=0.001,
    )
    if task == "ranking":
        kwargs["lm"] = "glove"
    cfg = get_configurations(kwargs)
    return cfg.model(), kwargs


def _enhance(graph: GraphJson, definition: dict[str, Any]) -> GraphJson:
    task = definition["task"]
    by_id = {node.id: node for node in graph.nodes}
    dataset = by_id["dataset_1"]
    dataset.component = "dataset.mind" if task == "matching" else "dataset.mindrb"
    from studio.backend.registry.registry import default_params

    if definition.get("quick_demo"):
        dataset.component = "dataset.mind-demo"
    dataset.params = default_params(dataset.component)
    user = by_id["user_1"]
    predictor = by_id["predictor_1"]
    item = by_id.get("item_1")

    if item:
        dataset.position = Position(x=40, y=260)
        item.position = Position(x=340, y=60)
        user.position = Position(x=340, y=340)
        predictor.position = Position(x=680, y=190)
    else:
        dataset.position = Position(x=40, y=240)
        user.position = Position(x=340, y=150)
        predictor.position = Position(x=680, y=240)

    training_component = "training.matching_task" if task == "matching" else "training.ranking_task"
    graph.nodes.append(
        GraphNode(id="training_1", component=training_component,
                  params=default_params(training_component), position=Position(x=1000, y=430))
    )
    graph.edges.append(GraphEdge(id="e_train_score", source=predictor.id, sourceHandle="score",
                                 target="training_1", targetHandle="score"))
    graph.edges.append(GraphEdge(id="e_train_dataset", source=dataset.id, sourceHandle="dataset",
                                 target="training_1", targetHandle="dataset"))

    hidden_size = 32 if definition.get("quick_demo") else 256
    graph.train = TrainConfig(hidden_size=hidden_size, item_hidden_size=hidden_size, metrics=_EVALS_BY_TASK[task], epochs=1)
    if definition.get("quick_demo"):
        graph.train.batch_size = 32
        for node in graph.nodes:
            for field in ("hidden_size", "item_hidden_size"):
                if field in node.params:
                    node.params[field] = hidden_size
    graph.graph_id = f"template_{definition['id']}"
    graph.name = definition["name"]
    graph.template_id = definition["id"]
    graph.task = task  # type: ignore[assignment]
    graph.description = definition["description"]
    return graph


@lru_cache(maxsize=1)
def get_templates() -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for definition in TEMPLATE_DEFS:
        path = REPO_ROOT / definition["yaml"]
        entry: dict[str, Any] = dict(definition)
        entry["source_yaml"] = definition["yaml"]
        if not path.is_file():
            entry["available"] = False
            entry["error"] = "Reference YAML not found in the repository."
            templates.append(entry)
            continue
        try:
            resolved, kwargs = _resolved_model(definition["yaml"], definition["task"])
            graph = import_yaml(yaml.safe_dump(resolved, sort_keys=False))
            graph = _enhance(graph, definition)
            entry.update(
                available=True,
                graph=graph.model_dump(mode="json"),
                resolved_config=resolved,
                load_kwargs={k: str(v) for k, v in kwargs.items()},
                verified_by="trainer.get_configurations",
            )
        except Exception as exc:  # noqa: BLE001
            # A broken/incomplete upstream YAML is reported honestly, never faked.
            entry["available"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        templates.append(entry)
    return templates


def get_template(template_id: str) -> dict[str, Any] | None:
    return next((t for t in get_templates() if t["id"] == template_id), None)
