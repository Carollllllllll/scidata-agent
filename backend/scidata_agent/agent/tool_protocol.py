from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ToolStatus = Literal["completed", "partial", "failed", "skipped"]


class ToolSpec(BaseModel):
    """Discoverable capability contract for an Agent tool.

    The contract is deliberately local and dependency-free. It mirrors the
    information an MCP client needs to discover a tool, while allowing the
    current process to execute existing Python implementations directly.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    artifact_types: frozenset[str] = frozenset()
    requires_artifact: bool = False
    requires_local_path: bool = False
    global_action: bool = False
    side_effects: frozenset[str] = frozenset()
    supports_retry: bool = True
    supports_resume: bool = True
    can_run_parallel: bool = False

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        value = " ".join(str(value).split())
        if not value:
            raise ValueError("tool name must not be empty")
        return value


class ToolCall(BaseModel):
    """A model-authored request to invoke one registered tool."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    purpose: str = ""
    priority: Literal["high", "medium", "low"] = "medium"
    gap_ids: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    workflow_revision: int = 0
    idempotency_key: str | None = None

    def effective_idempotency_key(self) -> str:
        """Return a stable key that ignores explanatory text from the model."""
        if self.idempotency_key:
            return self.idempotency_key
        payload = {
            "tool_name": self.tool_name,
            "arguments": _canonicalize(self.arguments),
            "workflow_revision": max(0, int(self.workflow_revision)),
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ToolResult(BaseModel):
    """Uniform, auditable result envelope returned by every tool call."""

    call_id: str
    tool_name: str
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    retry_count: int = 0
    workflow_revision: int = 0
    idempotency_key: str | None = None
    cached: bool = False


def _canonicalize(value: Any) -> Any:
    """Make nested tool arguments deterministic without changing their values."""
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_canonicalize(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def failed_tool_result(
    call: ToolCall,
    message: str,
    *,
    elapsed_ms: float = 0.0,
    retry_count: int = 0,
) -> ToolResult:
    return ToolResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        status="failed",
        errors=[message],
        elapsed_ms=max(0.0, float(elapsed_ms)),
        retry_count=max(0, int(retry_count)),
        workflow_revision=max(0, int(call.workflow_revision)),
        idempotency_key=call.effective_idempotency_key(),
    )


def timed_call(handler, call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
    """Invoke a handler and normalize a malformed handler exception boundary."""
    started = time.perf_counter()
    try:
        result = handler(call, context, options)
        if not isinstance(result, ToolResult):
            raise TypeError("tool handler must return ToolResult")
        if result.elapsed_ms <= 0:
            result.elapsed_ms = (time.perf_counter() - started) * 1000
        if result.idempotency_key is None:
            result.idempotency_key = call.effective_idempotency_key()
        return result
    except Exception as exc:
        return failed_tool_result(
            call,
            f"Tool {call.tool_name!r} failed: {exc}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
