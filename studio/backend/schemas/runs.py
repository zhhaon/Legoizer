"""Schemas for training/evaluation runs and environment diagnostics."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
RunKind = Literal["train", "evaluate"]


class CurvePoint(BaseModel):
    epoch: int
    step: int | None = None
    value: float
    metric: str = "loss"


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    kind: RunKind = "train"
    status: RunStatus = "pending"
    experiment_name: str = ""
    model_label: str = ""
    dataset: str = ""
    task: str = ""
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str = ""
    blockers: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    loss_curve: list[CurvePoint] = Field(default_factory=list)
    validation_curve: list[CurvePoint] = Field(default_factory=list)
    log_offset: int = 0
    log_lines: list[str] = Field(default_factory=list)
    log_file: str = ""
    result_file: str = ""
    signature: str = ""
    graph: dict[str, Any] | None = None
    model_yaml: str = ""


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: dict[str, Any]
    preflight_only: bool = False


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    graph: dict[str, Any] | None = None


class EnvironmentReport(BaseModel):
    python: str
    executable: str
    torch: str
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: list[dict[str, Any]] = Field(default_factory=list)
    device_name: str | None
    dependencies: dict[str, str]
    trainer_script: bool
    data_root_exists: bool
    datasets: list[dict[str, Any]]
    embeddings: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
