"""Pydantic schemas for the component registry wire format."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "dataset",
    "feature",
    "item_encoder",
    "user_encoder",
    "predictor",
    "training",
    "evaluation",
]

ParamKind = Literal["integer", "number", "boolean", "string", "enum", "list", "object", "number_or_list", "string_or_list"]


class PortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    required: bool = False
    description: str = ""


class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParamKind = "string"
    default: Any = None
    required: bool = False
    nullable: bool = False
    read_only: bool = False
    default_hint: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] = Field(default_factory=list)
    description: str = ""
    source: Literal["introspected", "curated", "dataset"] = "introspected"
    config_path: str = ""
    advanced: bool = False


class ComponentSpec(BaseModel):
    """Everything the frontend needs; it never has to know a Python path."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    category: Category
    role: str
    description: str
    icon: str = "Box"
    status: Literal["verified", "available", "conceptual"] = "available"
    executable: bool = True
    conceptual: bool = False
    backend_class: str | None = None
    meta_value: str | None = None
    inputs: list[PortSpec] = Field(default_factory=list)
    outputs: list[PortSpec] = Field(default_factory=list)
    parameters: list[ParamSpec] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=lambda: ["matching", "ranking"])
    source: dict[str, Any] = Field(default_factory=dict)
    detail: str = ""
    status_note: str = ""


class MetricSpec(BaseModel):
    name: str
    argument: Literal["cutoff", "threshold"] | None = None


class ComponentCatalog(BaseModel):
    components: list[ComponentSpec]
    categories: list[dict[str, Any]]
    metrics: list[MetricSpec]
    port_types: list[dict[str, Any]]
    generated_from: str
