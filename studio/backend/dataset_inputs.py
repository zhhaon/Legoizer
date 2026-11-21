"""Resolve the item columns shown in Studio and used by Legommenders."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from refconfig import CType, RefConfig

from studio.backend.config import EMBED_CONFIG_DIR, REPO_ROOT
from studio.backend.registry.registry import get_component
from studio.backend.resource_catalog import EXTERNAL_HF_COMPONENT, EXTERNAL_HF_VOCAB, LM_MODELS, dataset_processor, item_checkpoint, resource_args
from studio.backend.schemas.graph import GraphJson


def dataset_settings(graph: GraphJson) -> tuple[str, str, str | None]:
    dataset = next((node for node in graph.nodes if node.component.startswith("dataset.")), None)
    if dataset is None:
        return "config/data/mind.yaml", "config/embed/null.yaml", None
    spec = get_component(dataset.component)
    allowed = {p.name: p for p in (spec.parameters if spec else [])}
    data_config = str(dataset.params.get("data_config", allowed["data_config"].default if "data_config" in allowed else "config/data/mind.yaml"))
    embed_config = str(dataset.params.get("embed_config", allowed["embed_config"].default if "embed_config" in allowed else "config/embed/null.yaml"))
    lm = dataset.params.get("lm") or (allowed["lm"].default if "lm" in allowed else None)
    external_item, checkpoint = item_resource_owner(graph)
    if checkpoint:
        lm = LM_MODELS[checkpoint][0]
    lm = {"opt": "optbase"}.get(lm, lm) or "glove"
    embed_config = ("config/embed/bertbase.yaml" if lm == "bert" else
                    "config/embed/llama.yaml" if lm.startswith("llama") else
                    "config/embed/opt.yaml" if lm.startswith("opt") else
                    "config/embed/glm.yaml" if lm == "glm" else "config/embed/glove.yaml")
    # Server-controlled allow-list: never accept arbitrary paths from the client.
    if not _path_allowed(data_config, REPO_ROOT / "config/data") and not _path_allowed(data_config, REPO_ROOT / "config/recbench"):
        data_config = "config/data/mind.yaml"
    if (any(item_resource_owner(graph)) or dataset_external_config(graph) is not None
            or not _path_allowed(embed_config, EMBED_CONFIG_DIR)):
        embed_config = "config/embed/null.yaml"
    return data_config, embed_config, str(lm) if lm else None


def dataset_variables(graph: GraphJson, data_config: str, lm: str | None) -> dict[str, str]:
    dataset = next((node for node in graph.nodes if node.component.startswith("dataset.")), None)
    lang = dataset.params.get("lang") if dataset else None
    if data_config == "config/data/xmind.yaml" and (not dataset or "lm" not in dataset.params):
        lm = "llama1"
    return resource_args(data_config, lm, lang)


def _path_allowed(relative: str, allowed_dir: Path) -> bool:
    try:
        candidate = (REPO_ROOT / relative).resolve()
        return candidate.is_relative_to(allowed_dir.resolve()) and candidate.is_file()
    except (OSError, ValueError):
        return False


def input_entries(values: list) -> list[dict[str, Any]]:
    return [
        {"column": next(iter(value)), "truncate": next(iter(value.values()))}
        if isinstance(value, dict) else {"column": value, "truncate": None}
        for value in values
    ]


def default_item_inputs(values: list, embed: dict, checkpoint: str | None) -> list:
    selected = embed.get("embeddings") or []
    vocab = LM_MODELS[checkpoint][0] if checkpoint else (selected[0].get("vocab_name") if len(selected) == 1 else None)
    if not vocab:
        return values
    converted = []
    for entry in input_entries(values):
        column, truncate = entry["column"], entry["truncate"]
        if column == "item_id" or "->" in column:
            if not checkpoint:
                converted.append({column: truncate} if truncate is not None else column)
            continue
        if checkpoint or "@" in column:
            column = column.split("@", 1)[0] + f"@{vocab}"
        converted.append({column: truncate} if truncate is not None else column)
    return converted or [{f"title@{vocab}": 30}]


def source_column(column: str) -> str:
    return re.split(r"-{1,2}>", column, maxsplit=1)[0].strip()


def item_resource_owner(graph: GraphJson) -> tuple[bool, str | None]:
    item = next((get_component(node.component) for node in graph.nodes if node.component.startswith("item.")), None)
    return (item is not None and item.id == EXTERNAL_HF_COMPONENT,
            item_checkpoint(item.meta_value or "") if item else None)


def dataset_external_config(graph: GraphJson) -> dict[str, Any] | None:
    external_item, checkpoint = item_resource_owner(graph)
    dataset = next((node for node in graph.nodes if node.component.startswith("dataset.")), None)
    if external_item or checkpoint or not dataset or dataset.params.get("token_source") != "external_hf":
        return None
    fields = {
        "model_id": "", "revision": "main", "tokenizer_id": "", "tokenizer_revision": "",
        "freeze_token_embeddings": True,
    }
    return {"encoding_mode": "token_embeddings", **{
        field: dataset.params.get("external_" + field, default) for field, default in fields.items()
    }}


def _external_defaults(values: list, *, keep_other_columns: bool) -> list:
    converted = []
    has_text = False
    for entry in input_entries(values):
        column, truncate = entry["column"], entry["truncate"]
        source = source_column(column)
        if "@" in source:
            column = column.replace(source, source.split("@", 1)[0] + f"@{EXTERNAL_HF_VOCAB}", 1)
            has_text = True
        elif not keep_other_columns:
            continue
        converted.append({column: truncate} if truncate is not None else column)
    if not has_text:
        converted.append({f"title@{EXTERNAL_HF_VOCAB}": 30})
    return converted


def _available_columns(data: dict, processor: str, defaults: list, checkpoint: str | None) -> list[str]:
    columns = {source_column(entry["column"]) for entry in input_entries(defaults)}
    item_path = (REPO_ROOT / data["item"]["ut"]).resolve()
    if item_path.is_relative_to(REPO_ROOT):
        try:
            features = json.loads((item_path / "meta.json").read_text())["features"]
            columns.update(feature["name"] for feature in features)
        except FileNotFoundError:
            pass
    columns.add(data["column_map"]["item_col"])
    if processor in {"mind", "mind-demo", "oncemind", "xmind"}:
        vocabs = {entry["column"].rsplit("@", 1)[1] for entry in input_entries(defaults) if "@" in entry["column"]}
        columns.update(f"{attr}@{vocab}" for attr in ("prompt", "prompt_title", "title", "prompt_abstract", "abstract", "prompt_category", "category", "prompt_subcategory", "subcategory") for vocab in vocabs)
        if processor != "xmind":
            columns.update(("category", "subcategory"))
    active_vocabs = {entry["column"].rsplit("@", 1)[1] for entry in input_entries(defaults) if "@" in entry["column"]}
    columns = {column for column in columns if "@" not in column or column.rsplit("@", 1)[1] in active_vocabs}
    if checkpoint:
        vocab = LM_MODELS[checkpoint][0]
        columns = {column for column in columns if column.endswith(f"@{vocab}")}
    return sorted(columns)


def resolve_dataset_inputs(graph: GraphJson, *, use_override: bool = True) -> dict[str, Any]:
    data_config, embed_config, lm = dataset_settings(graph)
    variables = dataset_variables(graph, data_config, lm)
    config = RefConfig().add(CType.SMART, data=data_config, embed=embed_config, **variables).parse()
    if lm and lm.startswith("opt"):
        for entry in config["embed"].get("embeddings") or []:
            entry["vocab_name"] = lm
    return resolve_item_inputs(graph, config["data"], config["embed"], data_config, use_override=use_override)


def resolve_item_inputs(
    graph: GraphJson, data: dict, embed: dict, data_config: str, *, use_override: bool = True
) -> dict[str, Any]:
    dataset = next((node for node in graph.nodes if node.component.startswith("dataset.")), None)
    external_item, checkpoint = item_resource_owner(graph)
    external_dataset = dataset_external_config(graph) is not None
    raw = data["item"]["inputs"]
    defaults = (_external_defaults(raw, keep_other_columns=external_dataset)
                if external_item or external_dataset else default_item_inputs(raw, embed, checkpoint))
    columns = _available_columns(data, dataset_processor(data_config), defaults, checkpoint)
    if external_item:
        columns = [column for column in columns if column.endswith(f"@{EXTERNAL_HF_VOCAB}")]
    elif external_dataset:
        columns = [column for column in columns if "@" not in column or column.endswith(f"@{EXTERNAL_HF_VOCAB}")]
    override = dataset.params.get("item_inputs") if dataset and use_override else None
    if isinstance(override, list) and all(isinstance(entry, dict) and isinstance(entry.get("column"), str) for entry in override):
        preferred = {source_column(entry["column"]).split("@", 1)[0]: source_column(entry["column"]) for entry in input_entries(defaults)}
        for column in sorted(columns):
            preferred.setdefault(column.split("@", 1)[0], column)
        override = [{**entry, "column": preferred.get(source_column(entry["column"]).split("@", 1)[0], entry["column"])} for entry in override]
    inputs = defaults if override is None else _custom_inputs(override, columns)
    if (external_item or external_dataset) and not any(
        source_column(entry["column"]).endswith(f"@{EXTERNAL_HF_VOCAB}") for entry in input_entries(inputs)
    ):
        raise ValueError("Select at least one text column for the external tokenizer.")
    return {
        "defaults": input_entries(defaults), "columns": columns,
        "inputs": input_entries(inputs), "values": inputs,
    }


def _custom_inputs(value: Any, columns: list[str]) -> list:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("Dataset inputs need between 1 and 64 columns.")
    inputs, names = [], set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"column", "truncate"}:
            raise ValueError("Each dataset input needs a column and optional truncation.")
        column, truncate = entry["column"], entry["truncate"]
        if not isinstance(column, str) or not re.fullmatch(r"[\w@.-]+(?: +-> +[\w@.-]+)?", column):
            raise ValueError("Choose an item column or use 'column -> alias'.")
        source = source_column(column)
        name = column.split("->", 1)[-1].strip()
        if source not in columns:
            raise ValueError(f"Item column {source!r} is unavailable for this dataset and encoder.")
        if name in names or (name != source and name in columns):
            raise ValueError(f"Item input name {name!r} is already used.")
        if truncate is not None and (type(truncate) is not int or truncate < 1):
            raise ValueError("Item input truncation must be a positive integer.")
        names.add(name)
        inputs.append({column: truncate} if truncate is not None else column)
    return inputs
