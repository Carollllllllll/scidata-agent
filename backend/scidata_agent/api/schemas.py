from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["queued", "running", "completed", "partial", "failed", "cancelled"]


class TaskProgress(BaseModel):
    current: int | None = None
    total: int | None = None


class TaskSubmissionResponse(BaseModel):
    task_id: str
    status: TaskStatus
    status_url: str
    events_url: str


class TaskResponse(BaseModel):
    """Stable lifecycle envelope used for every task state."""

    task_id: str
    status: TaskStatus
    research_question: str | None = None
    current_step: str | None = None
    message: str | None = None
    progress: TaskProgress | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error: dict[str, Any] | None = None
    uploads: list[dict[str, Any]] = Field(default_factory=list)
    event: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    download_urls: dict[str, str] = Field(default_factory=dict)
    review_decisions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse] = Field(default_factory=list)
    count: int = 0


class TaskEventsResponse(BaseModel):
    task_id: str
    status: TaskStatus
    events: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    qwen_configured: bool
    model: str
    vl_model: str
    text_model_pool: list[str] | tuple[str, ...]
    vl_model_pool: list[str] | tuple[str, ...]
    cors_origins: list[str]
    agent_loop: list[str]


class ReviewRequest(BaseModel):
    decision: Literal["approved", "needs_changes", "rejected"]
    note: str | None = Field(default=None, max_length=2000)


class ReviewDecision(BaseModel):
    record_id: str
    review_id: str | None = None
    subject_id: str | None = None
    subject_type: str | None = None
    decision: Literal["approved", "needs_changes", "rejected"]
    note: str | None = None
    updated_at: str
