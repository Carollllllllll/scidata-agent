from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scidata_agent.agent.tool_protocol import ToolCall


class AgentDecision(BaseModel):
    """Structured next-step decision produced from an Agent observation."""

    decision: Literal["continue", "stop"] = "continue"
    reason: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    stop_reason: str | None = None


class PolicyDecision(BaseModel):
    """Result of deterministic validation of a model-authored decision."""

    allowed: bool
    tool_calls: list[ToolCall] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class StopDecision(BaseModel):
    """Deterministic decision on whether the Agent may terminate."""

    allowed: bool
    reasons: list[str] = Field(default_factory=list)
