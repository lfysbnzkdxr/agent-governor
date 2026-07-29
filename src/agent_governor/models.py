"""Pydantic data models for Agent Governor."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a unit of work to be executed."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    tenant_id: str = ""
    model: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    created_at: float = Field(default_factory=time.time)


class TaskResult(BaseModel):
    """Result of a task execution."""

    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    content: str = ""
    model: str = ""
    usage: TokenUsage = Field(default_factory=lambda: TokenUsage())
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: float = Field(default_factory=time.time)


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def model_post_init(self, __context: Any) -> None:
        if self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)


class CallRecord(BaseModel):
    """Audit record for a single LLM call."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = Field(default_factory=time.time)
    caller: str = ""
    tenant_id: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of a quality gate validation."""

    passed: bool
    gate_name: str = ""
    score: float = 1.0
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class DagNodeStatus(StrEnum):
    """Status of a DAG node during execution."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DagNodeResult(BaseModel):
    """Result of a single DAG node execution."""

    node_id: str
    status: DagNodeStatus = DagNodeStatus.PENDING
    result: TaskResult | None = None
    validation: ValidationResult | None = None
    attempts: int = 0
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


class DagExecutionResult(BaseModel):
    """Result of a full DAG execution."""

    graph_id: str
    status: TaskStatus = TaskStatus.PENDING
    node_results: dict[str, DagNodeResult] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error: str | None = None


# Rebuild models to resolve forward references
TaskResult.model_rebuild()
DagNodeResult.model_rebuild()
