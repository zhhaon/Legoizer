"""Model Compiler: Graph JSON -> real Legommenders YAML.

Pipeline::

    Graph JSON -> Graph Validator -> Component Registry -> Model Compiler
              -> Legommenders YAML -> trainer.get_configurations() verification

Only keys that exist in ``LegoConfig``, in a real ``config_class`` signature or
in a shipped YAML are ever emitted.  Original template files are never edited.
"""
from __future__ import annotations
from studio.backend.i18n import localized


import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from studio.backend.compiler.archetypes import detect
from studio.backend.compiler.verify import verify_with_legommenders
from studio.backend.config import (
    GENERATED_DIR,
    MODEL_CONFIG_DIR,
    REPO_ROOT,
)
from studio.backend.registry.registry import component_index, get_component
from studio.backend.schemas.graph import (
    CompileResult,
    GraphEdge,
    GraphJson,
    GraphNode,
    Position,
    TrainConfig,
    ValidationReport,
)
from studio.backend.dataset_inputs import dataset_external_config, dataset_settings, dataset_variables
from studio.backend.validator.validator import validate
from studio.backend.resource_catalog import EXTERNAL_HF_COMPONENT, LM_MODELS, item_checkpoint

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _nested_set(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            return
    cursor[parts[-1]] = value


def _section_params(node: GraphNode, section: str) -> dict[str, Any]:
    """Extract only the parameters whose ``config_path`` targets ``section``."""
    spec = get_component(node.component)
    if not spec:
        return {}
    prefix = f"config.{section}."
    result: dict[str, Any] = {}
    for param in spec.parameters:
        if not param.config_path.startswith(prefix):
            continue
        if param.name not in node.params:
            continue
        key = param.config_path[len(prefix):]
        _nested_set(result, key, node.params[param.name])
    return result


def _scalar(node: GraphNode | None, name: str, default: Any) -> Any:
    if node is None or name not in node.params:
        return default
    value = node.params[name]
    return default if value is None else value


def role_nodes(graph: GraphJson, role: str) -> list[GraphNode]:
    index = component_index()
    return [n for n in graph.nodes if (index.get(n.component) or None) and index[n.component].role == role]


def _archetype(graph: GraphJson):
    item = next(iter(role_nodes(graph, "item_encoder")), None)
    user = next(iter(role_nodes(graph, "user_encoder")), None)
    predictor = next(iter(role_nodes(graph, "predictor")), None)
    return detect(
        task=graph.task,
        item_component=item.component if item else None,
        user_component=user.component if user else None,
        predictor_component=predictor.component if predictor else None,
    )


def build_model_config(graph: GraphJson) -> tuple[dict[str, Any], list[str]]:
    index = component_index()
    item = next(iter(role_nodes(graph, "item_encoder")), None)
    user = next(iter(role_nodes(graph, "user_encoder")), None)
    predictor = next(iter(role_nodes(graph, "predictor")), None)
    training = next(iter(role_nodes(graph, "training")), None)
    notes: list[str] = []
    if not user or not predictor:
        raise ValueError("A User Encoder and a Predictor are required for compilation")

    archetype = _archetype(graph)
    hidden = graph.train.hidden_size
    item_hidden = graph.train.item_hidden_size
    checkpoint = item_checkpoint(index[item.component].meta_value or "") if item else None
    if checkpoint:
        item_hidden = LM_MODELS[checkpoint][1]
    elif item:
        configured = item.params.get("item_hidden_size")
        if isinstance(configured, int):
            item_hidden = configured
    item_hidden = item_hidden or hidden
    graph.train.item_hidden_size = item_hidden

    use_item_content = item is not None
    use_neg_sampling = graph.task == "matching"
    config: dict[str, Any] = {
        "use_item_content": use_item_content,
        "hidden_size": hidden,
        "item_hidden_size": item_hidden,
        "use_fast_eval": bool(_scalar(training, "use_fast_eval", _scalar(predictor, "use_fast_eval", True))),
        "use_neg_sampling": use_neg_sampling,
    }
    if use_neg_sampling:
        config["neg_count"] = int(_scalar(training, "neg_count", 4))
    if item:
        page_size = item.params.get("item_page_size")
        if isinstance(page_size, int) and page_size > 0:
            config["item_page_size"] = page_size
        elif checkpoint:
            config["item_page_size"] = 64
        item_config = _section_params(item, "item_config")
        if item_config:
            config["item_config"] = item_config
    user_config = _section_params(user, "user_config")
    if user_config:
        config["user_config"] = user_config
    predictor_config = _section_params(predictor, "predictor_config")
    if predictor_config:
        config["predictor_config"] = predictor_config

    meta: dict[str, str] = {
        "user": index[user.component].meta_value or "",
        "predictor": index[predictor.component].meta_value or "",
    }
    if item:
        meta["item"] = index[item.component].meta_value or ""

    name = archetype.label if archetype.key else "CustomModel"
    if item and item.component == EXTERNAL_HF_COMPONENT:
        notes.append(localized('backend.2678ada3e627', "Training resolves the external Hugging Face revision and token embedding width before constructing the model.", {}))
    elif dataset_external_config(graph) is not None:
        notes.append(localized('backend.5c0ea0a0b90f', "Training resolves the Dataset's external tokenizer and pretrained token embeddings before preprocessing.", {}))
    notes.extend(archetype.notes)
    return {"name": name, "meta": meta, "config": config}, notes


def build_exp_config(graph: GraphJson) -> dict[str, Any]:
    metrics = list(graph.train.metrics)
    if graph.train.dev_metric not in metrics:
        metrics.insert(0, graph.train.dev_metric)
    return {
        "load": {"sign": None, "model_only": True, "strict": False},
        "store": {"metric": graph.train.dev_metric, "patience": graph.train.patience},
        "policy": {
            "epoch": graph.train.epochs,
            "lr": graph.train.lr,
            "item_lr": None,
            "freeze_emb": False,
            "pin_memory": False,
            "epoch_batch": 0,
            "batch_size": graph.train.batch_size,
            "accumulate_batch": graph.train.accumulate_batch,
            "device": "cpu" if graph.train.device == "cpu" else "gpu",
            "n_warmup": 0,
            "check_interval": -2,
            "simple_dev": bool(graph.train.simple_dev),
            "disable_tqdm": False,
        },
        "metrics": metrics,
    }


def _write_generated(graph: GraphJson, model_config: dict[str, Any], exp_config: dict[str, Any], artifact_id: str | None) -> tuple[Path, Path]:
    slug = _SAFE.sub("_", graph.graph_id)[:48] or uuid.uuid4().hex[:8]
    folder = GENERATED_DIR / (artifact_id or f"{slug}-{uuid.uuid4().hex[:8]}")
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.yaml"
    exp_path = folder / "exp.yaml"
    model_path.write_text(yaml.safe_dump(model_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    exp_path.write_text(yaml.safe_dump(exp_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return model_path, exp_path


def compile_graph(graph: GraphJson, *, write_files: bool = True, artifact_id: str | None = None) -> CompileResult:
    report: ValidationReport = validate(graph)
    if not report.valid:
        raise ValueError("Graph is not valid: " + "; ".join(e.message for e in report.errors[:5]))

    model_config, notes = build_model_config(graph)
    exp_config = build_exp_config(graph)
    data_config, embed_config, lm = dataset_settings(graph)
    model_path, exp_path = _write_generated(graph, model_config, exp_config, artifact_id) if write_files else (Path(""), Path(""))
    yaml_text = yaml.safe_dump(model_config, sort_keys=False, allow_unicode=True)

    verification: dict[str, Any] = {}
    if write_files:
        verification = verify_with_legommenders(
            model_path=str(model_path.relative_to(REPO_ROOT)),
            data_config=data_config,
            exp_config=str(exp_path.relative_to(REPO_ROOT)),
            embed_config=embed_config,
            hidden_size=graph.train.hidden_size,
            batch_size=graph.train.batch_size,
            lr=graph.train.lr,
            variables=dataset_variables(graph, data_config, lm),
        )
        notes.extend(verification.get("checks", []))
        for error in verification.get("errors", []):
            notes.append(localized('backend.5ae247af6cfe', f"Verification error: {error}", {'p0': f'{error}'}))

    archetype = _archetype(graph)
    return CompileResult(
        yaml_text=yaml_text,
        model_dict=model_config,
        model_label=archetype.label,
        exact_template=archetype.reference_yaml,
        reference_yaml=archetype.reference_yaml,
        notes=notes,
        note_translations=[getattr(note, "reference", None) for note in notes],
        generated_file=str(model_path.relative_to(REPO_ROOT)) if write_files else None,
        data_config=data_config,
        exp_config=str(exp_path.relative_to(REPO_ROOT)) if write_files else None,
        embed_config=embed_config,
        validation=report,
        compatible_with_reference=bool(verification.get("configs_instantiated", write_files is False)),
    )


# --------------------------------------------------------------------------- #
# Reverse direction: import a Legommenders model YAML back into Graph JSON.
# --------------------------------------------------------------------------- #
def _resolve_yaml_imports(raw: dict[str, Any]) -> dict[str, Any]:
    imports = raw.pop("$$import", [])
    if isinstance(imports, str):
        imports = [imports]
    merged: dict[str, Any] = {}
    for relative in imports:
        path = (MODEL_CONFIG_DIR / str(relative)).resolve()
        if not path.is_relative_to(MODEL_CONFIG_DIR.resolve()) or not path.is_file():
            continue
        child = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, _resolve_yaml_imports(child))
    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def import_yaml(yaml_text: str) -> GraphJson:
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("The YAML must describe a mapping (name / meta / config).")
    data = _resolve_yaml_imports(raw)
    meta = data.get("meta") or {}
    config = data.get("config") or {}
    index = component_index()

    def find_component(meta_value: str | None, role: str) -> str | None:
        if not meta_value or "${" in str(meta_value):
            return None
        wanted = str(meta_value).lower()
        for spec in index.values():
            if spec.role == role and (spec.meta_value or "").lower() == wanted:
                return spec.id
        return None

    task = "matching" if config.get("use_neg_sampling", True) else "ranking"
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    dataset_id = "dataset_1"
    nodes.append(GraphNode(id=dataset_id, component="dataset.mind", position=Position(x=40, y=200)))
    item_id = user_id = predictor_id = None

    item_component = find_component(meta.get("item"), "item_encoder")
    if item_component:
        item_id = "item_1"
        nodes.append(GraphNode(id=item_id, component=item_component, position=Position(x=320, y=60)))
        edges.append(GraphEdge(id="e_content", source=dataset_id, sourceHandle="content",
                               target=item_id, targetHandle="content"))
    user_component = find_component(meta.get("user"), "user_encoder") or "user.pooling"
    user_id = "user_1"
    nodes.append(GraphNode(id=user_id, component=user_component, position=Position(x=320, y=300)))
    predictor_component = find_component(meta.get("predictor"), "predictor") or "predictor.dot"
    predictor_id = "predictor_1"
    nodes.append(GraphNode(id=predictor_id, component=predictor_component, position=Position(x=640, y=180)))

    raw_hidden = config.get("hidden_size", 256)
    hidden = raw_hidden if isinstance(raw_hidden, int) and not isinstance(raw_hidden, bool) else 256
    for node in nodes:
        spec = get_component(node.component)
        if spec:
            node.params = {p.name: p.default for p in spec.parameters if p.default is not None}
        if node.id in (item_id, user_id, predictor_id):
            node.params["hidden_size"] = hidden

    def apply_section(node_id: str, section: str, values: dict[str, Any]) -> None:
        node = next(n for n in nodes if n.id == node_id)
        spec = get_component(node.component)
        if not spec:
            return
        prefix = f"config.{section}."
        for param in spec.parameters:
            if not param.config_path.startswith(prefix):
                continue
            key = param.config_path[len(prefix):]
            candidate: Any = None
            if key in values:
                candidate = values[key]
            elif key.startswith("inputer_config.") and isinstance(values.get("inputer_config"), dict):
                candidate = values["inputer_config"].get(key.split(".", 1)[1])
            if candidate is not None and not (isinstance(candidate, str) and "${" in candidate):
                node.params[param.name] = candidate

    if item_id:
        apply_section(item_id, "item_config", config.get("item_config") or {})
        item_node = next(n for n in nodes if n.id == item_id)
        if isinstance(config.get("item_hidden_size"), int):
            item_node.params["item_hidden_size"] = config["item_hidden_size"]
    apply_section(user_id, "user_config", config.get("user_config") or {})
    apply_section(predictor_id, "predictor_config", config.get("predictor_config") or {})

    edges.append(GraphEdge(id="e_user", source=user_id, sourceHandle="user",
                           target=predictor_id, targetHandle="user"))
    if item_id:
        edges.append(GraphEdge(id="e_history", source=item_id, sourceHandle="history",
                               target=user_id, targetHandle="history"))
        edges.append(GraphEdge(id="e_candidate", source=item_id, sourceHandle="candidate",
                               target=predictor_id, targetHandle="candidate"))
    else:
        edges.append(GraphEdge(id="e_history", source=dataset_id, sourceHandle="history",
                               target=user_id, targetHandle="history"))
        edges.append(GraphEdge(id="e_features", source=dataset_id, sourceHandle="features",
                               target=predictor_id, targetHandle="features"))
    edges.append(GraphEdge(id="e_dataset", source=dataset_id, sourceHandle="features",
                           target=user_id, targetHandle="features"))

    return GraphJson(
        graph_id=f"imported_{uuid.uuid4().hex[:8]}",
        name=str(data.get("name") or "Imported Model"),
        task=task,  # type: ignore[arg-type]
        description="Imported from a Legommenders YAML configuration.",
        nodes=nodes,
        edges=edges,
        train=TrainConfig(hidden_size=hidden, item_hidden_size=hidden),
    )
