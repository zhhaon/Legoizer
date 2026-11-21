"""Graph Validator.

Checks connection semantics, structure, dimensions, task compatibility and
configuration values *before* anything is compiled or executed.  It reuses the
framework's own task-compatibility flags (``BasePredictor.allow_matching`` /
``allow_ranking``) that ``model.lego_config.LegoConfig.build_components`` uses.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable

from studio.backend.i18n import localized, references
from studio.backend.compiler.archetypes import detect
from studio.backend.config import REPO_ROOT
from studio.backend.dataset_inputs import dataset_external_config, item_resource_owner, resolve_dataset_inputs
from studio.backend.resource_catalog import EXTERNAL_HF_COMPONENT, external_hf_sources
from studio.backend.registry.model_params import ACTIVATIONS
from studio.backend.registry.registry import (
    component_index,
    get_component,
    port_compatible,
)
from studio.backend.schemas.components import ComponentSpec, ParamSpec
from studio.backend.schemas.graph import (
    GraphEdge,
    GraphJson,
    GraphNode,
    ValidationIssue,
    ValidationReport,
    ValidationSummary,
)

EXECUTABLE_ROLES = {"dataset", "item_encoder", "user_encoder", "predictor", "training", "evaluation"}


def issue(
    code: str,
    severity: str,
    message: str,
    *,
    reason: str = "",
    fix: str = "",
    node_id: str | None = None,
    edge_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        reason=reason,
        fix=fix,
        node_id=node_id,
        edge_id=edge_id,
        translations=references(message=message, reason=reason, fix=fix),
    )


class _GraphContext:
    def __init__(self, graph: GraphJson):
        self.graph = graph
        self.index = component_index()
        self.by_role: dict[str, list[GraphNode]] = defaultdict(list)
        self.specs: dict[str, ComponentSpec] = {}
        self.nodes: dict[str, GraphNode] = {node.id: node for node in graph.nodes}
        for node in graph.nodes:
            spec = self.index.get(node.component)
            if spec is None:
                continue
            self.specs[node.id] = spec
            self.by_role[spec.role].append(node)

    def first(self, role: str) -> GraphNode | None:
        nodes = self.by_role.get(role) or []
        return nodes[0] if nodes else None

    def out_port(self, node: GraphNode, handle: str):
        spec = self.specs.get(node.id)
        if not spec:
            return None
        return next((p for p in spec.outputs if p.name == handle), None)

    def in_port(self, node: GraphNode, handle: str):
        spec = self.specs.get(node.id)
        if not spec:
            return None
        return next((p for p in spec.inputs if p.name == handle), None)


def _check_type(value: Any, param: ParamSpec) -> bool:
    if param.type == "number_or_list":
        values = value if isinstance(value, list) else [value]
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x) for x in values)
    if param.type == "string_or_list":
        return all(isinstance(x, str) for x in (value if isinstance(value, list) else [value]))
    if param.type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if param.type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    if param.type == "boolean":
        return isinstance(value, bool)
    if param.type == "string":
        return isinstance(value, str)
    if param.type == "enum":
        return isinstance(value, str) and (not param.choices or value in param.choices)
    if param.type == "list":
        return isinstance(value, (list, tuple))
    if param.type == "object":
        return isinstance(value, dict)
    return True


def _validate_params(ctx: _GraphContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node in ctx.graph.nodes:
        spec = ctx.specs.get(node.id)
        if spec is None:
            continue
        declared = {p.name: p for p in spec.parameters}
        for name, value in node.params.items():
            param = declared.get(name)
            if param is None:
                issues.append(
                    issue(
                        "unknown_parameter",
                        "warning",
                        localized('validation.unknown_parameter.message.00c83cef', f"Node `{spec.name}` has unknown parameter `{name}`.", {'p0': f'{spec.name}', 'p1': f'{name}'}),
                        reason=localized('validation.unknown_parameter.reason.32dd7b16', "The component registry has no such configuration key; it will be ignored by the compiler.", {}),
                        fix=localized('validation.unknown_parameter.fix.6d7d21f4', "Remove the parameter or choose a component that declares it.", {}),
                        node_id=node.id,
                    )
                )
                continue
            if param.read_only or (value is None and param.nullable):
                continue
            if not _check_type(value, param):
                issues.append(
                    issue(
                        "parameter_type",
                        "error",
                        localized('validation.parameter_type.message.fd0fbd02', f"Parameter `{name}` of `{spec.name}` must be {param.type}, got {type(value).__name__}.", {'p0': f'{name}', 'p1': f'{spec.name}', 'p2': f'{param.type}', 'p3': f'{type(value).__name__}'}),
                        reason=localized('validation.parameter_type.reason.c3d44021', f"Registry schema for {node.component}.{name} declares type {param.type}.", {'p0': f'{node.component}', 'p1': f'{name}', 'p2': f'{param.type}'}),
                        fix=localized('validation.parameter_type.fix.4dd7e921', f"Set `{name}` to a valid {param.type} value.", {'p0': f'{name}', 'p1': f'{param.type}'}),
                        node_id=node.id,
                    )
                )
                continue
            values = value if isinstance(value, list) else [value]
            if param.type in ("integer", "number", "number_or_list"):
                if any((param.minimum is not None and entry < param.minimum) or
                       (param.maximum is not None and entry > param.maximum) for entry in values):
                    lower = param.minimum if param.minimum is not None else "-∞"
                    upper = param.maximum if param.maximum is not None else "∞"
                    issues.append(issue("parameter_range", "error", localized('validation.parameter_range.message.6b3adba2', f"`{name}` must be within [{lower}, {upper}].", {'p0': f'{name}', 'p1': f'{lower}', 'p2': f'{upper}'}),
                                        node_id=node.id))
            if name.endswith("_hidden_units") or name == "hidden_units":
                if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in values):
                    issues.append(issue("parameter_range", "error", localized('validation.parameter_range.message.05900b79', f"`{name}` must contain positive integer widths.", {'p0': f'{name}'}),
                                        node_id=node.id))
            if param.type == "string_or_list" or name.endswith("_activation"):
                if any(entry not in ACTIVATIONS and entry.lower() not in {"relu", "sigmoid", "tanh"}
                       for entry in values):
                    issues.append(issue("parameter_choice", "error", localized('validation.parameter_choice.message.f67f71d2', f"`{name}` must use supported activation names.", {'p0': f'{name}'}),
                                        fix=localized('validation.parameter_choice.fix.6ad877be', "For example: ReLU, GELU, SiLU or Tanh.", {}), node_id=node.id))
            if param.choices and value not in param.choices:
                issues.append(
                    issue("parameter_choice", "error", localized('validation.parameter_choice.message.4bd46ec2', f"Parameter `{name}` has an unsupported value `{value}`.", {'p0': f'{name}', 'p1': f'{value}'}),
                          reason=localized('validation.parameter_choice.reason.f8e52f94', "The component accepts a fixed set of values.", {}),
                          fix=localized('validation.parameter_choice.fix.a37fb60e', f"Choose one of: {', '.join(param.choices[:6])}.", {'p0': f"{', '.join(param.choices[:6])}"}), node_id=node.id)
                )
        if node.component == EXTERNAL_HF_COMPONENT and node.params.get("model_id"):
            try:
                external_hf_sources(node.params)
            except ValueError as exc:
                issues.append(issue("external_hf_source", "error", str(exc), node_id=node.id))
        for name, param in declared.items():
            if param.required and (name not in node.params or node.params[name] in (None, "")):
                issues.append(
                    issue("missing_parameter", "error", localized('validation.missing_parameter.message.3e60f193', f"Required parameter `{name}` of `{spec.name}` is empty.", {'p0': f'{name}', 'p1': f'{spec.name}'}),
                          reason=localized('validation.missing_parameter.reason.23d5a2ac', "This field is required to build a valid LegoConfig.", {}),
                          fix=localized('validation.missing_parameter.fix.528613a2', f"Fill in `{name}` in the Properties panel (default: {param.default}).", {'p0': f'{name}', 'p1': f'{param.default}'}),
                          node_id=node.id)
                )
    return issues


def _validate_edges(ctx: _GraphContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    graph = ctx.graph
    seen: set[tuple[str, str, str, str]] = set()
    occupied_inputs: set[tuple[str, str]] = set()
    for edge in graph.edges:
        source = ctx.nodes.get(edge.source)
        target = ctx.nodes.get(edge.target)
        if source is None or target is None:
            issues.append(
                issue("dangling_edge", "error", localized('validation.dangling_edge.message.04079bd1', "An edge references a node that no longer exists.", {}),
                      reason=localized('validation.dangling_edge.reason.6c4bff76', f"source={edge.source}, target={edge.target}", {'p0': f'{edge.source}', 'p1': f'{edge.target}'}),
                      fix=localized('validation.dangling_edge.fix.e13ba8a3', "Delete the stale connection and reconnect the nodes.", {}), edge_id=edge.id)
            )
            continue
        if source.id == target.id:
            issues.append(
                issue("self_loop", "error", localized('validation.self_loop.message.c69cdf43', "A node cannot be connected to itself.", {}),
                      reason=localized('validation.self_loop.reason.b75b3405', "Legommenders components form a directed acyclic configuration graph.", {}),
                      fix=localized('validation.self_loop.fix.73a367fa', "Remove the self-connection.", {}), node_id=source.id, edge_id=edge.id)
            )
            continue
        out_port = ctx.out_port(source, edge.sourceHandle)
        in_port = ctx.in_port(target, edge.targetHandle)
        if out_port is None or in_port is None:
            issues.append(
                issue("invalid_port", "error", localized('validation.invalid_port.message.c4dffde4', "An edge is attached to a port that does not exist.", {}),
                      reason=localized('validation.invalid_port.reason.711a11db', f"{edge.source}.{edge.sourceHandle} → {edge.target}.{edge.targetHandle}", {'p0': f'{edge.source}', 'p1': f'{edge.sourceHandle}', 'p2': f'{edge.target}', 'p3': f'{edge.targetHandle}'}),
                      fix=localized('validation.invalid_port.fix.d5b624e5', "Reconnect using the ports declared by the component registry.", {}), edge_id=edge.id)
            )
            continue
        if not port_compatible(out_port.type, in_port.type):
            issues.append(
                issue(
                    "port_type_mismatch",
                    "error",
                    localized('validation.port_type_mismatch.message.e712e37a', f"Incompatible ports: `{source.component}` outputs {out_port.type}, "
                    f"but `{target.component}` expects {in_port.type} on `{in_port.name}`.", {'p0': f'{source.component}', 'p1': f'{out_port.type}', 'p2': f'{target.component}', 'p3': f'{in_port.type}', 'p4': f'{in_port.name}'}),
                    reason=localized('validation.port_type_mismatch.reason.4a8c2fd1', f"Port type {out_port.type} is not accepted by an input of type {in_port.type}.", {'p0': f'{out_port.type}', 'p1': f'{in_port.type}'}),
                    fix=localized('validation.port_type_mismatch.fix.63422d56', f"Connect a node producing one of: {', '.join(_accepted(in_port.type))}.", {'p0': f"{', '.join(_accepted(in_port.type))}"}),
                    node_id=target.id,
                    edge_id=edge.id,
                )
            )
        key = (edge.source, edge.sourceHandle, edge.target, edge.targetHandle)
        if key in seen:
            issues.append(
                issue("duplicate_edge", "error", localized('validation.duplicate_edge.message.56d891f7', "Duplicate connection detected.", {}),
                      reason=localized('validation.duplicate_edge.reason.159ae785', "The same output is wired twice into the same input.", {}),
                      fix=localized('validation.duplicate_edge.fix.a8239108', "Remove the duplicate edge.", {}), edge_id=edge.id)
            )
        seen.add(key)
        target_key = (edge.target, edge.targetHandle)
        if target_key in occupied_inputs:
            issues.append(
                issue("multiple_inputs", "error", localized('validation.multiple_inputs.message.19817c31', f"Input `{edge.targetHandle}` of `{target.component}` has more than one source.", {'p0': f'{edge.targetHandle}', 'p1': f'{target.component}'}),
                      reason=localized('validation.multiple_inputs.reason.4557adcf', "A required module input has exactly one upstream producer in Legommenders.", {}),
                      fix=localized('validation.multiple_inputs.fix.66d3f670', "Keep a single incoming connection for this port.", {}), node_id=target.id, edge_id=edge.id)
            )
        occupied_inputs.add(target_key)
    return issues


def _accepted(port_type: str) -> list[str]:
    from studio.backend.registry.metadata import PORT_DEFINITIONS

    return PORT_DEFINITIONS.get(port_type, {}).get("accepts", [port_type])


def _has_cycle(graph: GraphJson) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        for child in adjacency.get(node, []):
            if visit(child):
                return True
        state[node] = 2
        return False

    return any(visit(node.id) for node in graph.nodes if state.get(node.id, 0) == 0)


def _validate_structure(ctx: _GraphContext) -> list[ValidationIssue]:
    graph, issues = ctx.graph, []
    if not graph.nodes:
        return [issue("empty_graph", "error", localized('validation.empty_graph.message.4baa1d82', "The model canvas is empty.", {}),
                      reason=localized('validation.empty_graph.reason.68843664', "There is nothing to compile.", {}), fix=localized('validation.empty_graph.fix.844fa976', "Load a template or drag components onto the canvas.", {}))]

    if _has_cycle(graph):
        issues.append(
            issue("cycle", "error", localized('validation.cycle.message.e6726f8f', "The graph contains a cycle.", {}),
                  reason=localized('validation.cycle.reason.eaf35a84', "Legommenders operators form a directed acyclic pipeline.", {}),
                  fix=localized('validation.cycle.fix.fdd013c5', "Remove the edge that closes the loop.", {}))
        )

    degree: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    for node in graph.nodes:
        spec = ctx.specs.get(node.id)
        if spec and spec.executable and degree[node.id] == 0 and len(graph.nodes) > 1:
            issues.append(
                issue("orphan_node", "error", localized('validation.orphan_node.message.4fa92249', f"`{spec.name}` is an isolated node.", {'p0': f'{spec.name}'}),
                      reason=localized('validation.orphan_node.reason.c99f48f7', "The node has no incoming or outgoing connections, so it cannot contribute to the model.", {}),
                      fix=localized('validation.orphan_node.fix.4a4e13e1', "Connect it to the pipeline, or delete it from the canvas.", {}), node_id=node.id)
            )

    # Required input ports.
    incoming: dict[tuple[str, str], int] = defaultdict(int)
    for edge in graph.edges:
        incoming[(edge.target, edge.targetHandle)] += 1
    for node in graph.nodes:
        spec = ctx.specs.get(node.id)
        if not spec:
            continue
        for port in spec.inputs:
            if port.required and incoming[(node.id, port.name)] == 0:
                issues.append(
                    issue("missing_input", "error", localized('validation.missing_input.message.6a0cf38f', f"`{spec.name}` is missing required input `{port.name}` ({port.type}).", {'p0': f'{spec.name}', 'p1': f'{port.name}', 'p2': f'{port.type}'}),
                          reason=localized('validation.missing_input.reason.e356f659', "The component cannot be instantiated without this upstream signal.", {}),
                          fix=localized('validation.missing_input.fix.b534f38f', f"Connect a node with a {port.type} output to `{port.name}`.", {'p0': f'{port.type}', 'p1': f'{port.name}'}), node_id=node.id)
                )

    # Cardinality of the fixed Legommenders slots.
    for role, label in (("dataset", "Dataset"), ("user_encoder", "User / Behavior Encoder"),
                        ("predictor", "Predictor")):
        nodes = ctx.by_role.get(role) or []
        if len(nodes) == 0:
            issues.append(
                issue("missing_component", "error", localized('validation.missing_component.message.62998bc6', f"The model has no {label} node.", {'p0': f'{label}'}),
                      reason=localized('validation.missing_component.reason.fd010575', "LegoConfig always builds a user operator and a predictor; the data config is required too.", {}),
                      fix=localized('validation.missing_component.fix.4f04a19b', f"Add one {label} component from the library.", {'p0': f'{label}'}))
            )
        elif len(nodes) > 1:
            issues.append(
                issue("duplicate_component", "error", localized('validation.duplicate_component.message.15c393b7', f"The model has {len(nodes)} {label} nodes.", {'p0': f'{len(nodes)}', 'p1': f'{label}'}),
                      reason=localized('validation.duplicate_component.reason.db4f8953', "Legommenders has exactly one slot for this module.", {}),
                      fix=localized('validation.duplicate_component.fix.8c88bb6a', "Remove the extra node(s).", {}), node_id=nodes[1].id)
            )

    item_nodes = ctx.by_role.get("item_encoder") or []
    if len(item_nodes) > 1:
        issues.append(issue("duplicate_component", "error", localized('validation.duplicate_component.message.dbaf9d55', "The model has more than one Item Encoder.", {}),
                            reason=localized('validation.duplicate_component.reason.4c6fbc65', "LegoConfig has a single optional item operator.", {}),
                            fix=localized('validation.duplicate_component.fix.425d4680', "Remove the extra Item Encoder node.", {}), node_id=item_nodes[1].id))
    # Content encoding and the training objective are independent choices.
    # Without an Item Encoder, the runtime looks up item ID embeddings.
    if len(ctx.by_role.get("training") or []) > 1:
        issues.append(issue("duplicate_component", "error", localized('validation.duplicate_component.message.c520a227', "More than one objective block is present.", {}),
                            reason=localized('validation.duplicate_component.reason.6ce153cb', "Only one training objective can drive config.use_neg_sampling.", {}),
                            fix=localized('validation.duplicate_component.fix.e33f9d5f', "Keep a single Matching/Ranking Task node.", {})))
    return issues


def _validate_tasks(ctx: _GraphContext) -> list[ValidationIssue]:
    graph, issues = ctx.graph, []
    predictor = ctx.first("predictor")
    if predictor:
        spec = ctx.specs[predictor.id]
        if graph.task not in spec.tasks:
            issues.append(
                issue(
                    "task_incompatible",
                    "error",
                    localized('validation.task_incompatible.message.6d930e7f', f"`{spec.name}` does not support the {graph.task} task.", {'p0': f'{spec.name}', 'p1': f'{graph.task}'}),
                    reason=(
                        localized('validation.task_incompatible.reason.883d923e', "LegoConfig.build_components() rejects use_neg_sampling=True for predictors with "
                        "allow_matching=False, and use_neg_sampling=False for predictors with allow_ranking=False.", {})
                    ),
                    fix=localized('validation.task_incompatible.fix.6ce625a5', "Switch the graph task, or replace the predictor with one that supports it.", {}),
                    node_id=predictor.id,
                )
            )
        if graph.task == "matching" and ctx.first("item_encoder") and "candidate" not in {e.targetHandle for e in graph.edges if e.target == predictor.id}:
            issues.append(
                issue("missing_input", "warning", localized('validation.missing_input.message.ffa3e19e', "The predictor has no candidate item input.", {}),
                      reason=localized('validation.missing_input.reason.40c13e13', "Matching scores B×(K+1) candidates; the connection documents that data flow.", {}),
                      fix=localized('validation.missing_input.fix.e7b12e3d', "Connect the Item Encoder `candidate` output to the predictor.", {}),
                      node_id=predictor.id)
            )
    for node in ctx.by_role.get("training") or []:
        spec = ctx.specs[node.id]
        if graph.task not in spec.tasks:
            issues.append(
                issue("task_incompatible", "error", localized('validation.task_incompatible.message.e52c1cbb', f"Objective `{spec.name}` requires a {spec.tasks[0]} task, but the graph is {graph.task}.", {'p0': f'{spec.name}', 'p1': f'{spec.tasks[0]}', 'p2': f'{graph.task}'}),
                      reason=localized('validation.task_incompatible.reason.97580608', "config.use_neg_sampling and the loss selected in model/legommender.py would disagree.", {}),
                      fix=localized('validation.task_incompatible.fix.1582bf04', f"Set the graph task to {spec.tasks[0]} or replace the objective node.", {'p0': f'{spec.tasks[0]}'}), node_id=node.id)
            )
    return issues


def _validate_dimensions(ctx: _GraphContext) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    runtime: list[ValidationIssue] = []
    global_hidden = ctx.graph.train.hidden_size
    for node in ctx.graph.nodes:
        spec = ctx.specs.get(node.id)
        if not spec or spec.role not in ("item_encoder", "user_encoder", "predictor"):
            continue
        local = node.params.get("hidden_size")
        if local is not None and isinstance(local, int) and local != global_hidden:
            errors.append(
                issue(
                    "dimension_mismatch",
                    "error",
                    localized('validation.dimension_mismatch.message.a1f7355e', f"Dimension Mismatch on `{spec.name}`: hidden_size={local}, but the run uses {global_hidden}.", {'p0': f'{spec.name}', 'p1': f'{local}', 'p2': f'{global_hidden}'}),
                    reason=(
                        localized('validation.dimension_mismatch.reason.b9bc3df9', f"All Legommenders operators share one LegoConfig.hidden_size. `{spec.name}` would output "
                        f"{local}-dimensional embeddings while the Dot predictor expects {global_hidden}.", {'p0': f'{spec.name}', 'p1': f'{local}', 'p2': f'{global_hidden}'})
                    ),
                    fix=localized('validation.dimension_mismatch.fix.84c6d992', f"Set `{spec.name}` hidden_size to {global_hidden} (or change Hidden Size in Training settings).", {'p0': f'{spec.name}', 'p1': f'{global_hidden}'}),
                    node_id=node.id,
                )
            )
    item = ctx.first("item_encoder")
    user = ctx.first("user_encoder")
    def value(node: GraphNode, name: str):
        spec = next((param for param in ctx.specs[node.id].parameters if param.name == name), None)
        if spec and spec.read_only:
            return spec.default
        return node.params.get(name, spec.default if spec else None)

    def check_heads(node: GraphNode, head_key: str, width: Any, label: str) -> None:
        heads = value(node, head_key)
        if type(heads) is int and heads > 0 and type(width) is int and width % heads:
            errors.append(issue("dimension_mismatch", "error",
                                localized('validation.dimension_mismatch.message.f7e114e0', f"{label} ({width}) must be divisible by {head_key} ({heads}).", {'p0': f'{label}', 'p1': f'{width}', 'p2': f'{head_key}', 'p3': f'{heads}'}),
                                fix=localized('validation.dimension_mismatch.fix.0e70ff5a', "Change the number of heads or the input width.", {}), node_id=node.id))

    item_width = ctx.graph.train.item_hidden_size or global_hidden
    if item:
        item_width = value(item, "item_hidden_size") or item_width
        check_heads(item, "num_attention_heads", item_width, "Item input width")
        global_proj = value(item, "global_proj_size")
        local_proj = value(item, "local_proj_size")
        if global_proj is not None and local_proj is not None and global_proj != local_proj:
            errors.append(issue("parameter_range", "error", localized('validation.parameter_range.message.19ce7318', "IISAN projection sizes must match when both are set.", {}),
                                node_id=item.id))
    if user and (not item or item.component != "item.cnncat"):
        user_width = item_width if not item or item.component == "item.ada" else global_hidden
        check_heads(user, "num_attention_heads", user_width, "User input width")
    predictor = ctx.first("predictor")
    if predictor and predictor.component == "predictor.autoint":
        check_heads(predictor, "num_attention_heads", value(predictor, "attention_dim"), "Attention width")
    if predictor and predictor.component == "predictor.finalmlp":
        for field in ("mlp1_hidden_units", "mlp2_hidden_units"):
            widths = value(predictor, field)
            if widths == []:
                errors.append(issue("parameter_range", "error", localized('validation.parameter_range.message.44ba0d23', f"{field} needs at least one layer.", {'p0': f'{field}'}),
                                    node_id=predictor.id))
            elif isinstance(widths, list) or widths is None:
                check_heads(predictor, "num_heads", widths[-1] if widths else global_hidden, field + " output")
    if item and user:
        if item.component == "item.cnncat":
            runtime.append(
                issue("runtime_dimension", "info",
                      localized('validation.runtime_dimension.message.ae4bc736', "CNNCat outputs hidden_size × number_of_item_columns dimensions.", {}),
                      reason=localized('validation.runtime_dimension.reason.c03d4fc6', "loader.manager.Manager feeds this output_dim into the user operator's input_dim.", {}),
                      fix=localized('validation.runtime_dimension.fix.a00036b7', "Confirm data.item.inputs column count at runtime.", {}), node_id=item.id)
            )
    if user and ctx.first("predictor"):
        runtime.append(
            issue("runtime_dimension", "info",
                  localized('validation.runtime_dimension.message.ff2d9c07', "User and candidate embedding dimensions are resolved by LegoConfig at model build time.", {}),
                  reason=localized('validation.runtime_dimension.reason.ee82fc66', "The exact tensor shapes depend on the loaded UniTok data; they cannot be inferred from the graph alone.", {}),
                  fix=localized('validation.runtime_dimension.fix.95233948', "Run the model-instantiation check (Generate Model → Verify) or start a real training run.", {}),
                  node_id=user.id)
        )
    if predictor:
        key = predictor.component.split(".", 1)[1]
        active_layers = []
        if key == "dcnv2":
            structure = value(predictor, "model_structure")
            if structure in ("stacked", "stacked_parallel"):
                active_layers.append(("stacked_dnn_hidden_units", 3, "dnn_dropout", "dnn_activations"))
            if structure in ("parallel", "stacked_parallel"):
                active_layers.append(("parallel_dnn_hidden_units", 3, "dnn_dropout", "dnn_activations"))
        elif key == "finalmlp":
            for prefix in ("mlp1", "mlp2"):
                active_layers.append((prefix + "_hidden_units", 3, prefix + "_dropout", prefix + "_hidden_activations"))
        elif key == "din":
            active_layers.extend([
                ("attention_hidden_units", 1, "attention_dropout"),
                ("dnn_hidden_units", 3, "net_dropout", "dnn_activations"),
            ])
        elif key == "masknet":
            active_layers.append(("hidden_units", 0, "dropout", "activations"))
        else:
            active_layers.append(("dnn_hidden_units", 0, "dnn_dropout", "dnn_activations"))
        for units, default_count, *settings in active_layers:
            widths = value(predictor, units)
            count = len(widths) if isinstance(widths, list) else default_count
            if count == 0 and key == "dcnv2":
                errors.append(issue("parameter_range", "error", localized('validation.parameter_range.message.44ba0d23', f"{units} needs at least one layer.", {'p0': f'{units}'}),
                                    node_id=predictor.id))
            for setting in settings:
                entries = value(predictor, setting)
                if isinstance(entries, list) and len(entries) < count:
                    errors.append(issue("parameter_range", "error",
                                        localized('validation.parameter_range.message.9404d0b8', f"{setting} needs at least {count} entries for {units}.", {'p0': f'{setting}', 'p1': f'{count}', 'p2': f'{units}'}),
                                        fix=localized('validation.parameter_range.fix.960fab6b', "Enter one value for all layers, or supply one value per layer.", {}),
                                        node_id=predictor.id))
    return errors, runtime


def _validate_resources(ctx: _GraphContext) -> list[ValidationIssue]:
    warnings: list[ValidationIssue] = []
    dataset = ctx.first("dataset")
    if dataset:
        rel = dataset.params.get("data_config")
        if isinstance(rel, str):
            path = (REPO_ROOT / rel).resolve()
            if not str(path).startswith(str(REPO_ROOT.resolve())) or not path.exists():
                warnings.append(issue("data_config_missing", "warning", localized('validation.data_config_missing.message.b3e9c6fe', f"Data configuration `{rel}` does not exist.", {'p0': f'{rel}'}),
                                      reason=localized('validation.data_config_missing.reason.05e9d3a4', "The client may only reference server-controlled config files.", {}),
                                      fix=localized('validation.data_config_missing.fix.8187612f', "Choose a data_config from the Properties panel list.", {}), node_id=dataset.id))
            else:
                data_paths = _read_data_paths(path)
                missing = [p for p in data_paths if not (REPO_ROOT / p).exists()]
                if missing:
                    warnings.append(
                        issue("dataset_files_missing", "warning",
                              localized('validation.dataset_files_missing.message.5a01fd2b', f"Dataset files are not available ({len(missing)} path(s) missing).", {'p0': f'{len(missing)}'}),
                              reason="Example: " + ", ".join(missing[:3]) + ". Train will try to prepare missing resources.",
                              fix=localized('validation.dataset_files_missing.fix.8e64fb24', "Click Train to download and prepare the dataset.", {}),
                              node_id=dataset.id)
                    )
        embed_rel = dataset.params.get("embed_config")
        if isinstance(embed_rel, str) and not any(item_resource_owner(ctx.graph)) and dataset_external_config(ctx.graph) is None:
            from studio.backend.registry.registry import _safe_read_yaml  # noqa: WPS437

            embed_path = (REPO_ROOT / embed_rel).resolve()
            embed_data = _safe_read_yaml(embed_path) or {}
            for entry in embed_data.get("embeddings") or []:
                if isinstance(entry, dict):
                    emb_path = entry.get("path")
                    if isinstance(emb_path, str) and "${" not in emb_path and not (REPO_ROOT / emb_path).exists():
                        warnings.append(
                            issue("embedding_missing", "warning", localized('validation.embedding_missing.message.a39cd5c3', f"Pre-trained embedding `{emb_path}` is missing.", {'p0': f'{emb_path}'}),
                                  reason=localized('validation.embedding_missing.reason.8c4b1827', "loader.embedding_hub loads this .npy file before building the model.", {}),
                                  fix=localized('validation.embedding_missing.fix.b6855948', "Click Train to prepare token embeddings, or choose another embedding configuration.", {}),
                                  node_id=dataset.id)
                        )
    for node in ctx.graph.nodes:
        spec = ctx.specs.get(node.id)
        if spec and spec.role == "item_encoder" and node.component.split(".")[-1].startswith(("bert", "llama", "glm", "opt")):
            warnings.append(
                issue("resource_required", "warning", localized('validation.resource_required.message.9e3a5aa4', f"`{spec.name}` needs cached language-model embeddings.", {'p0': f'{spec.name}'}),
                      reason=localized('validation.resource_required.reason.94e75bae', "The operator reads pre-computed embeddings generated by embed.py.", {}),
                      fix=localized('validation.resource_required.fix.7cb01fd8', "Generate them with `python embed.py --model <name>` and select the matching embed_config.", {}),
                      node_id=node.id)
            )
    return warnings


def _read_data_paths(path) -> list[str]:
    import re
    import yaml

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    except Exception:
        return []
    paths: list[str] = []
    base = data.get("base_dir")
    if isinstance(base, str) and "${" not in base:
        for section in ("item", "user"):
            ut = (data.get(section) or {}).get("ut")
            if isinstance(ut, str) and "${" not in ut:
                paths.append(ut)
        for key in ("train", "dev", "test"):
            value = (data.get("inter") or {}).get(key)
            if isinstance(value, str) and "${" not in value:
                paths.append(value)
    else:
        for match in re.findall(r"(data/[A-Za-z0-9_./-]+)", text):
            if "${" not in match:
                paths.append(match)
    return sorted(set(paths))


def validate(graph: GraphJson) -> ValidationReport:
    ctx = _GraphContext(graph)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    runtime: list[ValidationIssue] = []

    for node in graph.nodes:
        if node.component not in ctx.index:
            errors.append(
                issue("unknown_component", "error", localized('validation.unknown_component.message.49186d39', f"Unknown component `{node.component}` on node `{node.id}`.", {'p0': f'{node.component}', 'p1': f'{node.id}'}),
                      reason=localized('validation.unknown_component.reason.ec0420b6', "It is not present in the dynamic Component Registry.", {}),
                      fix=localized('validation.unknown_component.fix.ab4c5f3e', "Delete the node and replace it with a registered component.", {}), node_id=node.id)
            )

    errors.extend(_validate_params(ctx))
    dataset = ctx.first("dataset")
    dataset_external = dataset_external_config(graph)
    if dataset and dataset_external is not None:
        if ctx.first("item_encoder") is None:
            errors.append(issue(
                "external_hf_source", "error", localized('validation.external_hf_source.message.a558753c', "Dataset external token resources need an Item Encoder.", {}),
                node_id=dataset.id, fix=localized('validation.external_hf_source.fix.8cb42764', "Add an Item Encoder or select project token resources.", {}),
            ))
        try:
            external_hf_sources(dataset_external)
        except ValueError as exc:
            errors.append(issue("external_hf_source", "error", str(exc), node_id=dataset.id))
    if dataset and dataset.params.get("item_inputs") is not None and not any(
        entry.node_id == dataset.id and entry.severity == "error" for entry in errors
    ):
        try:
            resolve_dataset_inputs(graph)
        except (ValueError, KeyError, OSError) as exc:
            errors.append(issue("dataset_inputs", "error", str(exc), node_id=dataset.id))
    errors.extend(_validate_edges(ctx))
    errors.extend(_validate_structure(ctx))
    errors.extend(_validate_tasks(ctx))
    dim_errors, runtime = _validate_dimensions(ctx)
    errors.extend(dim_errors)
    warnings.extend(_validate_resources(ctx))

    item = ctx.first("item_encoder")
    user = ctx.first("user_encoder")
    predictor = ctx.first("predictor")
    dataset = ctx.first("dataset")
    archetype = detect(
        task=graph.task,
        item_component=item.component if item else None,
        user_component=user.component if user else None,
        predictor_component=predictor.component if predictor else None,
    )
    for note in archetype.notes:
        warnings.append(issue("model_identity", "info", note,
                              reason=localized('validation.model_identity.reason.26714af1', "Model identity is derived from the real component signature.", {}),
                              fix=localized('validation.model_identity.fix.320cb360', "Review the generated YAML before citing a paper.", {})))

    summary = ValidationSummary(
        nodes=len(graph.nodes),
        edges=len(graph.edges),
        task=graph.task,
        model_label=archetype.label,
        exact_template=archetype.reference_yaml,
        dataset=dataset.component if dataset else None,
        item_encoder=item.component if item else None,
        user_encoder=user.component if user else None,
        predictor=predictor.component if predictor else None,
    )
    # Validation stages may return mixed severities; only errors block execution.
    warnings.extend(entry for entry in errors if entry.severity != "error")
    errors = [entry for entry in errors if entry.severity == "error"]
    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=[w for w in warnings if w.severity == "warning"] + [w for w in warnings if w.severity == "info"],
        runtime_checks=runtime,
        summary=summary,
    )


def compatible_components_for(component_id: str, direction: str) -> list[dict[str, Any]]:
    """Components whose ports can connect to/from ``component_id``."""
    spec = get_component(component_id)
    if not spec:
        return []
    catalog = component_index()
    results: list[dict[str, Any]] = []
    for other in catalog.values():
        if other.id == component_id:
            continue
        matched: list[str] = []
        if direction == "downstream":
            for out in spec.outputs:
                for inp in other.inputs:
                    if port_compatible(out.type, inp.type):
                        matched.append(f"{out.name} ({out.type}) → {inp.name} ({inp.type})")
        else:
            for out in other.outputs:
                for inp in spec.inputs:
                    if port_compatible(out.type, inp.type):
                        matched.append(f"{out.name} ({out.type}) → {inp.name} ({inp.type})")
        if matched:
            results.append(
                {
                    "id": other.id,
                    "name": other.name,
                    "category": other.category,
                    "description": other.description,
                    "status": other.status,
                    "conceptual": other.conceptual,
                    "reasons": matched[:4],
                }
            )
    return results
