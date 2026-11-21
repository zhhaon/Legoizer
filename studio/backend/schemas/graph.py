"""Pydantic schemas for the unified Graph JSON format.

The TypeScript counterpart lives in ``studio/frontend/src/types/graph.ts``.
Every value that crosses the HTTP boundary is validated here; the compiler and
validator never receive an unvalidated ``dict``.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["matching", "ranking"]
ParamValue = Any  # validated against the registry parameter schema later


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    component: str = Field(min_length=1, max_length=128, description="Registry component id")
    type: str | None = Field(default=None, description="Node role, kept for Graph JSON compatibility")
    params: dict[str, ParamValue] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)
    label: str | None = None

    @field_validator("component")
    @classmethod
    def _canonical_training_component(cls, value: str) -> str:
        # Keep saved graphs compatible with the consolidated objective library.
        return {
            "training.bce_loss": "training.ranking_task",
            "training.cross_entropy": "training.matching_task",
            "training.negative_sampling": "training.matching_task",
        }.get(value, value)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        # ids are used in file names and in React keys; keep them boring.
        if not all(c.isalnum() or c in "_-." for c in value):
            raise ValueError("node id may only contain letters, digits, _ - .")
        return value


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    source: str
    sourceHandle: str
    target: str
    targetHandle: str


class TrainConfig(BaseModel):
    """Real, CLI-supported hyper-parameters (see ``trainer.get_configurations``)."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=64, ge=1, le=4096)
    accumulate_batch: int = Field(default=1, ge=1, le=4096)
    lr: float = Field(default=1e-3, gt=0, le=1)
    hidden_size: int = Field(default=256, ge=8, le=8192)
    item_hidden_size: int | None = Field(default=None, ge=8, le=16384)
    epochs: int = Field(default=1, ge=1, le=200)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    cuda_index: int | None = Field(default=None, ge=0)
    metrics: list[str] = Field(
        default_factory=lambda: ["GAUC", "MRR", "NDCG@1", "NDCG@5", "NDCG@10"]
    )
    dev_metric: str = "GAUC"
    patience: int = Field(default=5, ge=1, le=100)
    seed: int = Field(default=2023, ge=0)
    simple_dev: bool = False
    extra_cli: dict[str, str] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one evaluation metric is required")
        from utils.metrics import MetricPool

        return list(dict.fromkeys(str(MetricPool.make_metric(metric)) for metric in value))

    @field_validator("dev_metric")
    @classmethod
    def _dev_metric(cls, value: str) -> str:
        from utils.metrics import MetricPool

        return str(MetricPool.make_metric(value))



class GraphJson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_id: str = Field(default="model_001", max_length=128)
    name: str = Field(default="Custom Recommender", max_length=128)
    task: TaskType = "matching"
    description: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    train: TrainConfig = Field(default_factory=TrainConfig)
    template_id: str | None = None
    schema_version: str = "1.0"

    @model_validator(mode="after")
    def _remove_legacy_feature_annotations(self) -> GraphJson:
        # Feature blocks never affected compiled configuration or model execution.
        ids = {node.id for node in self.nodes if node.component.startswith("feature.")}
        if ids:
            self.nodes = [node for node in self.nodes if node.id not in ids]
            self.edges = [edge for edge in self.edges if edge.source not in ids and edge.target not in ids]
        return self

    @model_validator(mode="after")
    def _migrate_evaluation_nodes(self) -> GraphJson:
        from utils.metrics import MetricPool

        old = [node for node in self.nodes if node.component.startswith("evaluation.")]
        if not old:
            return self
        ids = {node.id for node in old}
        defaults = {"ndcg1": "NDCG@1", "ndcg5": "NDCG@5", "ndcg10": "NDCG@10",
                    "ndcg": "NDCG@10", "recall": "Recall@10", "hitratio": "HitRatio@10"}
        metrics = []
        for node in old:
            key = node.component.split(".", 1)[1]
            metric = node.params.get("metric") or defaults.get(key, key)
            if "k" in node.params and "@" in metric:
                metric = f"{metric.split('@')[0]}@{node.params['k']}"
            metrics.append(str(MetricPool.make_metric(metric)))
        self.train.metrics = list(dict.fromkeys(metrics))
        self.nodes = [node for node in self.nodes if node.id not in ids]
        self.edges = [edge for edge in self.edges if edge.source not in ids and edge.target not in ids]
        return self

    @field_validator("nodes")
    @classmethod
    def _unique_node_ids(cls, value: list[GraphNode]) -> list[GraphNode]:
        ids = [node.id for node in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node id")
        return value


class ValidationIssue(BaseModel):
    translations: dict[str, Any] = Field(default_factory=dict)

    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    reason: str = ""
    fix: str = ""
    node_id: str | None = None
    edge_id: str | None = None


class ValidationSummary(BaseModel):
    nodes: int = 0
    edges: int = 0
    task: str = "matching"
    model_label: str = "Custom Model"
    exact_template: str | None = None
    dataset: str | None = None
    item_encoder: str | None = None
    user_encoder: str | None = None
    predictor: str | None = None


class ValidationReport(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    runtime_checks: list[ValidationIssue] = Field(default_factory=list)
    summary: ValidationSummary = Field(default_factory=ValidationSummary)

    @property
    def all_issues(self) -> list[ValidationIssue]:
        return self.errors + self.warnings + self.runtime_checks


class CompileResult(BaseModel):
    note_translations: list[dict[str, Any] | None] = Field(default_factory=list)
    yaml_text: str
    model_dict: dict[str, Any]
    model_label: str
    exact_template: str | None = None
    reference_yaml: str | None = None
    notes: list[str] = Field(default_factory=list)
    generated_file: str | None = None
    data_config: str | None = None
    exp_config: str | None = None
    embed_config: str | None = None
    validation: ValidationReport
    compatible_with_reference: bool = True
