from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentTraceEvent(BaseModel):
    """One structured event in an Agent decision/tool trace."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:10]}")
    event_type: str
    iteration: int = 0
    call_id: str | None = None
    tool_name: str | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TraceRecorder:
    """In-memory trace collector; persistence is owned by the task checkpoint layer."""

    def __init__(self) -> None:
        self.events: list[AgentTraceEvent] = []

    def restore(self, events: list[dict[str, Any]] | None) -> int:
        """Restore valid events from a checkpoint without duplicating them."""
        restored = 0
        for raw in events or []:
            try:
                event = AgentTraceEvent.model_validate(raw)
            except Exception:
                continue
            self.events.append(event)
            restored += 1
        return restored

    def record(self, event_type: str, *, iteration: int = 0, **kwargs: Any) -> AgentTraceEvent:
        event = AgentTraceEvent(event_type=event_type, iteration=iteration, **kwargs)
        self.events.append(event)
        return event

    def record_decision(self, decision: Any, *, iteration: int = 0) -> AgentTraceEvent:
        return self.record(
            "agent_decision",
            iteration=iteration,
            status=getattr(decision, "decision", None),
            payload=decision.model_dump(mode="json"),
        )

    def record_tool_result(self, result: Any, *, iteration: int = 0) -> AgentTraceEvent:
        return self.record(
            "tool_result",
            iteration=iteration,
            call_id=getattr(result, "call_id", None),
            tool_name=getattr(result, "tool_name", None),
            status=getattr(result, "status", None),
            payload=result.model_dump(mode="json"),
        )

    def model_dump(self) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self.events]
