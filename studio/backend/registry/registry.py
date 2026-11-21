"""Dynamic Component Registry for Legoizer.

Discovery pipeline (nothing about a Python class is hard-coded in the frontend):

1. ``loader.class_hub.ClassHub`` scans ``model/operators`` and
   ``model/predictors`` and returns the *real* classes.
2. ``inspect.signature`` on each ``config_class.__init__`` (over the full MRO)
   yields real parameter names, annotations and defaults.
3. ``refconfig.jyonnie_config.parse`` reads every shipped model YAML (including
   ``$$import``) to learn which ``meta.*`` values exist, which role they are
   used in, and which config keys the project really accepts.
4. ``utils.metrics.MetricPool`` provides the real evaluation metrics.
5. A small curated overlay (``metadata.py``) adds prose, ports and ranges.

The result is exposed as Pydantic ``ComponentSpec`` objects via the API.
"""
from __future__ import annotations

import inspect
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from studio.backend.config import (
    DATA_CONFIG_DIR,
    EMBED_CONFIG_DIR,
    MODEL_CONFIG_DIR,
    RECBENCH_CONFIG_DIR,
    REPO_ROOT,
)
from studio.backend.registry import metadata as M
from studio.backend.registry.model_params import UNUSED_PARAMETERS, apply_model_constraints
from studio.backend.resource_catalog import EXTERNAL_HF_MODES, LM_BASE_CLASSES, LM_MODELS, TOKENIZERS, XMIND_LANGUAGES, item_checkpoint
from studio.backend.schemas.components import (
    ComponentCatalog,
    ComponentSpec,
    ParamSpec,
    PortSpec,
)

_PLACEHOLDER = re.compile(r"\$\{")
_META_TO_KEY = lambda value: value.replace("Operator", "").replace("Predictor", "").lower()  # noqa: E731

# Parameters supplied by LegoConfig/build_components, never by the user YAML.
_FRAMEWORK_KEYS = {"hidden_size", "input_dim", "inputer_config", "kwargs", "self", "args"}
_GLOBAL_KEYS_FROM_LEGO = {
    "item_hidden_size": 256,
    "item_page_size": 0,
    "use_fast_eval": True,
    "neg_count": 4,
}


def _safe_read_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML file and resolve ``$$import`` using the project's own parser."""
    try:
        from refconfig import jyonnie_config

        return jyonnie_config.parse(str(path)) or {}
    except Exception:
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None


_PLACEHOLDER_WITH_DEFAULT = re.compile(r"^\$\{[^:}]+:([^}]*)\}\$?$")


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and bool(_PLACEHOLDER.search(value))


def _placeholder_default(value: Any) -> Any:
    """Extract the real default from ``${name:default}$`` used by shipped YAMLs."""
    if not isinstance(value, str):
        return value
    match = _PLACEHOLDER_WITH_DEFAULT.match(value.strip())
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _scan_model_yamls() -> tuple[dict[str, set[str]], dict[str, dict[str, dict[str, Any]]], dict[str, list[str]]]:
    """Return (role_usage, observed_config_keys, yaml_files_per_key).

    role_usage:  class key -> {"item_encoder" | "user_encoder" | "predictor"}
    observed:    class key -> config section name -> {key: default value}
    yaml_files:  class key -> list of relative config paths
    """
    roles: dict[str, set[str]] = defaultdict(set)
    observed: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    files: dict[str, list[str]] = defaultdict(list)

    for path in sorted(MODEL_CONFIG_DIR.rglob("*.yaml")):
        data = _safe_read_yaml(path)
        if not isinstance(data, dict):
            continue
        meta = data.get("meta") or {}
        config = data.get("config") or {}
        rel = str(path.relative_to(REPO_ROOT))
        for meta_key, role, section in (
            ("item", "item_encoder", "item_config"),
            ("user", "user_encoder", "user_config"),
            ("predictor", "predictor", "predictor_config"),
        ):
            raw = meta.get(meta_key)
            if not isinstance(raw, str) or _is_placeholder(raw):
                continue
            key = _META_TO_KEY(raw)
            roles[key].add(role)
            if rel not in files[key]:
                files[key].append(rel)
            section_config = config.get(section)
            if isinstance(section_config, dict):
                for name, value in section_config.items():
                    if name == "inputer_config" and isinstance(value, dict):
                        for sub_name, sub_value in value.items():
                            resolved = sub_value if not _is_placeholder(sub_value) else _placeholder_default(sub_value)
                            if resolved is not None:
                                observed[key][section][f"inputer_config.{sub_name}"] = resolved
                        continue
                    resolved = value if not _is_placeholder(value) else _placeholder_default(value)
                    if resolved is not None:
                        observed[key][section][name] = resolved
    return roles, observed, files


def _signature_params(config_class: type) -> dict[str, inspect.Parameter]:
    """Collect constructor parameters across the whole config-class MRO."""
    params: dict[str, inspect.Parameter] = {}
    for klass in reversed(config_class.__mro__):
        if klass is object:
            continue
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        for name, parameter in sig.parameters.items():
            if name in _FRAMEWORK_KEYS:
                continue
            if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue
            params[name] = parameter
    return params


def _infer_kind(parameter: inspect.Parameter | None, value: Any) -> str:
    annotation = getattr(parameter, "annotation", inspect.Parameter.empty)
    if annotation is bool or isinstance(value, bool):
        return "boolean"
    if annotation is int or (isinstance(value, int) and not isinstance(value, bool)):
        return "integer"
    if annotation is float or isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _make_param(
    name: str,
    *,
    default: Any,
    kind: str,
    source: str,
    config_path: str,
    required: bool = False,
) -> ParamSpec:
    note = dict(M.COMMON_PARAM_NOTES.get(name, {}))
    return ParamSpec(
        name=name,
        type=note.get("type", kind),  # type: ignore[arg-type]
        default=default,
        required=required,
        minimum=note.get("minimum"),
        maximum=note.get("maximum"),
        choices=note.get("choices", []),
        description=note.get("description", f"Real configuration key `{name}`."),
        source=source,  # type: ignore[arg-type]
        config_path=config_path,
        advanced=bool(note.get("advanced", False)),
    )


def _python_components(
    roles: dict[str, set[str]],
    observed: dict[str, dict[str, dict[str, Any]]],
    files: dict[str, list[str]],
) -> list[ComponentSpec]:
    from loader.class_hub import ClassHub  # noqa: WPS433 - lazy heavy import

    # Legacy prototypes still use incompatible UniTok/inputer or predictor APIs.
    unsupported = {"semantic", "semanticmix", "poly"} | LM_BASE_CLASSES

    hubs = {
        "predictor": ClassHub.predictors(),
        "item_encoder": ClassHub.operators(),
        "user_encoder": ClassHub.operators(),
    }
    specs: list[ComponentSpec] = []

    for role, hub in hubs.items():
        for key, cls in sorted(hub.class_dict.items()):
            if inspect.isabstract(cls) or key in unsupported:
                continue
            assigned_roles = roles.get(key, set())
            if role != "predictor" and role not in assigned_roles | _infer_operator_role(key, cls):
                continue

            checkpoint = item_checkpoint(key) if role == "item_encoder" else None
            config_class = getattr(cls, "config_class", None)
            sig_params = _signature_params(config_class) if config_class else {}
            note = M.COMPONENT_NOTES.get(key, {})
            rel_yamls = files.get(key, [])[:6]
            section = {"item_encoder": "item_config", "user_encoder": "user_config", "predictor": "predictor_config"}[role]

            params: list[ParamSpec] = [
                _make_param(
                    "hidden_size",
                    default=256,
                    kind="integer",
                    source="curated",
                    config_path="global.hidden_size",
                    required=True,
                )
            ]
            observed_params = observed.get(key, {}).get(section, {})
            for name, parameter in sorted(sig_params.items()):
                if name in UNUSED_PARAMETERS.get(key, set()):
                    continue
                required = parameter.default is inspect.Parameter.empty
                default = observed_params.get(name) if required else parameter.default
                extra = note.get("params", {}).get(name, {})
                spec = _make_param(
                    name, default=default, kind=_infer_kind(parameter, default),
                    source="curated" if required and default is not None else "introspected",
                    config_path=f"config.{section}.{name}", required=required and default is None,
                )
                if key == "externalhf" and name == "encoding_mode":
                    spec.type = "enum"
                    spec.choices = list(EXTERNAL_HF_MODES)
                spec.nullable = parameter.default is None or name == "tune_from"
                spec.description = extra.get("description", spec.description)
                spec.advanced = bool(extra.get("advanced", spec.advanced))
                apply_model_constraints(spec, key, checkpoint)
                params.append(spec)

            inputer_name = getattr(getattr(cls, "inputer_class", None), "__name__", "")
            if key != "externalhf" and inputer_name in {"ConcatInputer", "FlattenSeqInputer"}:
                token_keys = ["use_cls_token", "use_sep_token"]
                if inputer_name == "FlattenSeqInputer":
                    token_keys.append("use_attr_sep_token")
                for key_name in token_keys:
                    param_name = f"inputer_config.{key_name}"
                    default = observed_params.get(param_name, key_name == "use_attr_sep_token")
                    params.append(_make_param(
                        param_name, default=default, kind="boolean", source="curated",
                        config_path=f"config.{section}.{param_name}",
                    ))

            if role == "item_encoder":
                width = _make_param(
                    "item_hidden_size", default=LM_MODELS[checkpoint][1] if checkpoint else None,
                    kind="integer", source="curated", config_path="config.item_hidden_size",
                )
                width.minimum, width.maximum = 1, None
                width.nullable = checkpoint is None
                width.read_only = checkpoint is not None or key == "externalhf"
                width.default_hint = "Matches hidden_size" if not checkpoint else None
                if key == "externalhf":
                    width.default_hint = "Detected from the pretrained token embedding matrix"
                    width.description = "Train detects the external model's input embedding width."
                if checkpoint:
                    width.description = "Fixed token embedding width of the selected checkpoint."
                page_size = _make_param(
                    "item_page_size", default=64 if checkpoint or key == "externalhf" else 0,
                    kind="integer", source="curated", config_path="config.item_page_size",
                )
                page_size.minimum, page_size.maximum = 0, None
                if checkpoint:
                    page_size.description = "Items per checkpoint encoding page. 0 uses the Studio default of 64."
                params.extend((width, page_size))
            if role == "predictor":
                params.append(
                    _make_param("use_fast_eval", default=True, kind="boolean", source="curated", config_path="config.use_fast_eval")
                )

            flags = {
                "allow_matching": getattr(cls, "allow_matching", None),
                "allow_ranking": getattr(cls, "allow_ranking", None),
                "keep_input_dim": getattr(cls, "keep_input_dim", None),
                "flatten_mode": getattr(cls, "flatten_mode", None),
            }
            tasks = ["matching", "ranking"]
            if role == "predictor":
                tasks = [t for t, ok in (("matching", flags["allow_matching"]), ("ranking", flags["allow_ranking"])) if ok is not False]

            component_id = f"{'item' if role == 'item_encoder' else 'user' if role == 'user_encoder' else 'predictor'}.{key}"
            display = note.get("name") or _prettify(key)
            description = note.get("description") or (
                f"Legommenders {role.replace('_', ' ')} backed by `{cls.__name__}`."
            )
            detail = note.get("detail") or (
                f"Python class: {cls.__module__}.{cls.__name__}. "
                f"Configuration class: {config_class.__name__ if config_class else 'n/a'}. "
            )
            verified = role in assigned_roles
            status_note = (
                "Referenced by shipped Legommenders YAML configurations."
                if verified
                else "Not referenced by any shipped YAML; verify resources before training."
            )
            if key == "externalhf":
                status_note = "Train downloads and validates the external Hugging Face checkpoint and tokenizer."
            if checkpoint:
                status_note += f" Train prepares the {checkpoint} tokenizer and pretrained weights."
            if role == "predictor" and flags["allow_matching"] is False:
                status_note += " Ranking only (allow_matching=False)."
            if role == "predictor" and flags["allow_ranking"] is False:
                status_note += " Matching only (allow_ranking=False)."

            specs.append(
                ComponentSpec(
                    id=component_id,
                    name=display,
                    category=role,  # type: ignore[arg-type]
                    role=role,
                    description=description,
                    icon=note.get("icon", "Box"),
                    status="verified" if verified else "available",
                    executable=True,
                    conceptual=False,
                    backend_class=f"{cls.__module__}.{cls.__name__}",
                    meta_value=_meta_value(key, cls.__name__, role),
                    inputs=_ports_for(role, "inputs"),
                    outputs=_ports_for(role, "outputs"),
                    parameters=params,
                    tasks=tasks or ["matching", "ranking"],
                    source={
                        "module": cls.__module__,
                        "class": cls.__name__,
                        "config_class": config_class.__name__ if config_class else None,
                        "signature": str(inspect.signature(config_class.__init__)) if config_class else "",
                        "yaml_files": rel_yamls,
                        "flags": flags,
                        "checkpoint": checkpoint,
                        "tokenizer": LM_MODELS[checkpoint][0] if checkpoint else None,
                        "discovered_by": "loader.class_hub.ClassHub",
                    },
                    detail=detail,
                    status_note=status_note,
                )
            )
    return specs


def _infer_operator_role(key: str, cls: type) -> set[str]:
    if getattr(cls, "flatten_mode", False):
        return {"user_encoder"}
    user_only = ("polyattention", "nullconcat", "nullsimple", "scsimple", "scflatten", "semantic")
    if key in user_only:
        return {"user_encoder"}
    if key == "pooling":
        return {"item_encoder", "user_encoder"}
    if key in ("attention", "transformer", "fastformer", "ada", "gru"):
        return {"item_encoder", "user_encoder"}
    return {"item_encoder"}


def _meta_value(key: str, class_name: str, role: str) -> str:
    """Reproduce the ``meta.*`` string that ClassHub resolves back to the class."""
    suffix = "Predictor" if role == "predictor" else "Operator"
    if class_name.endswith(suffix):
        return class_name[: -len(suffix)]
    return class_name.replace("Operator", "").replace("Predictor", "")


def _prettify(key: str) -> str:
    return key.replace("_", " ").title()


def _ports_for(role: str, direction: str) -> list[PortSpec]:
    common: dict[str, dict[str, list[dict[str, Any]]]] = {
        "item_encoder": {
            "inputs": [{"name": "content", "type": "ItemContent", "required": True}],
            "outputs": [
                {"name": "embedding", "type": "ItemEmbedding"},
                {"name": "history", "type": "ItemEmbeddingSequence"},
                {"name": "candidate", "type": "CandidateItemEmbedding"},
            ],
        },
        "user_encoder": {
            "inputs": [
                {"name": "history", "type": "ItemEmbeddingSequence", "required": True},
                {"name": "features", "type": "FeatureEmbedding"},
            ],
            "outputs": [{"name": "user", "type": "UserEmbedding"}],
        },
        "predictor": {
            "inputs": [
                {"name": "user", "type": "UserEmbedding", "required": True},
                {"name": "candidate", "type": "CandidateItemEmbedding"},
                {"name": "features", "type": "FeatureEmbedding"},
            ],
            "outputs": [{"name": "score", "type": "PredictionScore"}],
        },
    }
    return [PortSpec(**port) for port in common[role][direction]]


# --------------------------------------------------------------------------- #
# Datasets, features, training and evaluation
# --------------------------------------------------------------------------- #
def _dataset_description(key: str, label: str) -> str:
    descriptions = {
        "mind": "Microsoft news recommendation dataset with news content and user click histories.",
        "mind-demo": "A small, fixed MIND sample for quick training and evaluation.",
        "mind-lm": "MIND news recommendation with language model token inputs.",
        "mind-lm-prompt": "MIND news recommendation with prompt-based language model inputs.",
        "oncemind": "MIND news recommendation with ONCE impression splits.",
        "xmind": "Multilingual MIND news recommendation with translated news content.",
        "mindrb-with-id": "MIND RecBench data combining news content and item ID features.",
        "mindrb-dual-emb": "MIND RecBench data with dual pretrained token embeddings.",
    }
    return descriptions.get(key, f"{label.replace(' (RecBench)', '')} recommendation data with item features and user interaction histories.")


def _dataset_components() -> list[ComponentSpec]:
    configs: list[tuple[str, Path, dict[str, Any]]] = []
    for directory in (DATA_CONFIG_DIR, RECBENCH_CONFIG_DIR):
        for path in sorted(directory.glob("*.yaml")):
            if path.stem.startswith("base"):
                continue
            data = _safe_read_yaml(path)
            if not isinstance(data, dict):
                continue
            # Trainer data configs always declare interactions + a column map.
            if "inter" not in data or "column_map" not in data:
                continue
            configs.append((str(path.relative_to(REPO_ROOT)), path, data))

    all_paths = [rel for rel, _, _ in configs]
    embed_choices = [
        str(p.relative_to(REPO_ROOT)) for p in sorted(EMBED_CONFIG_DIR.glob("*.yaml"))
    ]
    specs: list[ComponentSpec] = []
    for rel, path, data in configs:
        stem = path.stem
        if path.parent == RECBENCH_CONFIG_DIR:
            key = f"{stem}rb" if not stem.endswith("rb") else stem
        else:
            key = stem
        raw_name = data.get("name", key)
        name = key if _is_placeholder(raw_name) else str(raw_name)
        label = M.DATASET_LABELS.get(key, M.DATASET_LABELS.get(name, _prettify(key)))
        params = [
            ParamSpec(
                name="data_config",
                type="enum",
                default=rel,
                required=True,
                choices=all_paths,
                description="Server-controlled Legommenders data YAML. The client cannot point to arbitrary files.",
                source="dataset",
                config_path="data_config",
            ),
            ParamSpec(
                name="embed_config",
                type="enum",
                default="config/embed/glove.yaml" if "glove" in yaml.safe_dump(data) else "config/embed/null.yaml",
                choices=embed_choices,
                description="Pre-trained embedding YAML passed as --embed to trainer.py.",
                source="dataset",
                config_path="embed_config",
            ),
        ]
        params.append(ParamSpec(
            name="item_inputs", type="list", default=None, nullable=True,
            description="Optional override of the dataset YAML item.inputs.", source="dataset",
            config_path="data.item.inputs",
        ))
        params.append(
            ParamSpec(
                name="lm",
                type="enum",
                default="llama1" if key == "xmind" else "glove",
                choices=[*TOKENIZERS, "opt"],
                description="Tokenizer for dataset configurations with selectable text inputs.",
                source="dataset",
                config_path="lm",
            )
        )
        params.append(ParamSpec(
            name="token_source", type="enum", default="project",
            choices=["project", "external_hf"], source="dataset",
            description="Select project presets or a matching tokenizer and embedding matrix from Hugging Face.",
        ))
        for name, kind, default, description in (
            ("model_id", "string", "", "Hugging Face repository supplying the pretrained input embedding matrix."),
            ("revision", "string", "main", "Model branch, tag, or commit hash."),
            ("tokenizer_id", "string", "", "Optional Hugging Face tokenizer repository. Leave empty to use the embedding model's tokenizer."),
            ("tokenizer_revision", "string", "", "Optional tokenizer branch, tag, or commit hash."),
            ("freeze_token_embeddings", "boolean", True, "Keep the imported token embedding weights fixed during training."),
        ):
            params.append(ParamSpec(
                name="external_" + name, type=kind, default=default,
                description=description, source="dataset",
            ))
        if key == "xmind":
            params.append(ParamSpec(
                name="lang", type="enum", default="cmn", choices=list(XMIND_LANGUAGES),
                description="Language of the translated xMIND news articles.", source="dataset",
            ))
        if key == "mind-demo":
            label = "MIND Demo (quick run)"
        specs.append(
            ComponentSpec(
                id=f"dataset.{key}",
                name=label,
                category="dataset",
                role="dataset",
                description=_dataset_description(key, label),
                icon="Database",
                status="verified",
                executable=True,
                conceptual=False,
                meta_value=name,
                inputs=[],
                outputs=[
                    PortSpec(name="dataset", type="Dataset"),
                    PortSpec(name="content", type="ItemContent"),
                    PortSpec(name="history", type="ItemEmbeddingSequence"),
                    PortSpec(name="features", type="FeatureEmbedding"),
                ],
                parameters=params,
                tasks=["matching", "ranking"],
                source={"yaml": rel, "data_name": name, "discovered_by": "config/data + config/recbench scan"},
                detail=(
                    "Discovered by scanning config/data/*.yaml and config/recbench/*.yaml and "
                    "resolving $$import with refconfig. The original YAML is never modified."
                ),
                status_note=(
                    "Train prepares missing resources. ONCE uses the configured impression split when available."
                    if key == "oncemind" else "Train prepares missing resources using the dataset publisher's files."
                ),
            )
        )
    return specs


def _training_components() -> list[ComponentSpec]:
    specs: list[ComponentSpec] = []
    for item in M.TRAINING_COMPONENTS:
        params: list[ParamSpec] = []
        if item["key"] == "matching_task":
            params.append(
                ParamSpec(
                    name="neg_count",
                    type="integer",
                    default=4,
                    minimum=1,
                    description="Number of negatives (LegoConfig.neg_count).",
                    source="curated",
                    config_path="config.neg_count",
                )
            )
        params.append(
            ParamSpec(
                name="use_fast_eval",
                type="boolean",
                default=True,
                description="Enable cached fast evaluation (LegoConfig.use_fast_eval).",
                source="curated",
                config_path="config.use_fast_eval",
            )
        )
        specs.append(
            ComponentSpec(
                id=f"training.{item['key']}",
                name=item["name"],
                category="training",
                role="training",
                description=item["description"],
                icon="Zap",
                status="verified",
                executable=True,
                conceptual=False,
                meta_value=item["key"],
                inputs=[
                    PortSpec(name="score", type="PredictionScore"),
                    PortSpec(name="dataset", type="Dataset"),
                ],
                outputs=[
                    PortSpec(name="config", type="TrainingConfig"),
                    PortSpec(name="run", type="TrainingData"),
                ],
                parameters=params,
                tasks=[item["task"]],
                source={"objective": item["task"], "mapped_to": "config.use_neg_sampling"},
                detail=item["detail"],
                status_note=f"Objective block — sets graph.task = {item['task']}.",
            )
        )
    return specs


@lru_cache(maxsize=1)
def get_catalog() -> ComponentCatalog:
    from utils.metrics import MetricPool

    roles, observed, files = _scan_model_yamls()
    components = (
        _python_components(roles, observed, files)
        + _dataset_components()
        + _training_components()
    )
    order = {category["id"]: index for index, category in enumerate(M.CATEGORY_META)}
    components.sort(key=lambda c: (order.get(c.category, 99), c.name.lower()))
    port_types = [
        {"id": key, **value} for key, value in M.PORT_DEFINITIONS.items()
    ]
    return ComponentCatalog(
        components=components,
        categories=[category for category in M.CATEGORY_META if category["id"] != "evaluation"],
        metrics=[
            {"name": metric.name,
             "argument": {"NDCG": "cutoff", "Recall": "cutoff", "HitRatio": "cutoff", "F1": "threshold"}.get(metric.name)}
            for metric in MetricPool.metric_list
        ],
        port_types=port_types,
        generated_from="loader.class_hub.ClassHub + config/model/**/*.yaml + utils.metrics.MetricPool",
    )


@lru_cache(maxsize=1)
def component_index() -> dict[str, ComponentSpec]:
    return {component.id: component for component in get_catalog().components}


def get_component(component_id: str) -> ComponentSpec | None:
    return component_index().get(component_id)


def port_compatible(output_type: str, input_type: str) -> bool:
    """True when an output of ``output_type`` may feed an input of ``input_type``."""
    definition = M.PORT_DEFINITIONS.get(input_type)
    if not definition:
        return False
    return output_type in definition["accepts"]


def default_params(component_id: str) -> dict[str, Any]:
    component = get_component(component_id)
    if not component:
        return {}
    return {p.name: p.default for p in component.parameters if p.default is not None}
