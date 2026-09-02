from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from scidata_agent.agent.decision import AgentDecision
from scidata_agent.agent.observation import AgentObservation, ObservationBuilder
from scidata_agent.agent.policy import AgentPolicy
from scidata_agent.agent.stop_gate import StopGate
from scidata_agent.agent.tool_protocol import ToolResult
from scidata_agent.agent.tool_registry import ToolRegistry
from scidata_agent.agent.tool_runtime import ToolRuntime
from scidata_agent.agent.trace import TraceRecorder


DecisionProvider = Callable[[AgentObservation, Any], AgentDecision]
ResultApplier = Callable[[ToolResult, Any], None]
IterationCallback = Callable[[int, list[ToolResult], Any], None]
TraceCallback = Callable[[dict[str, Any], Any], None]
CheckpointCallback = Callable[[Any], None]


class HarnessResult(BaseModel):
    """Summary returned by one generic Agent decision loop."""

    status: Literal["completed", "partial", "failed"]
    iterations: int = 0
    decisions: list[AgentDecision] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    stop_reason: str | None = None
    stop_rejections: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    terminal: bool = False


class AgentHarness:
    """Observe, decide, validate, execute, and evaluate an Agent context."""

    def __init__(
        self,
        registry: ToolRegistry,
        decision_provider: DecisionProvider,
        *,
        runtime: ToolRuntime | None = None,
        observation_builder: ObservationBuilder | None = None,
        policy: AgentPolicy | None = None,
        stop_gate: StopGate | None = None,
        result_applier: ResultApplier | None = None,
        after_iteration: IterationCallback | None = None,
        trace_callback: TraceCallback | None = None,
        checkpoint_callback: CheckpointCallback | None = None,
        policy_retry_limit: int = 2,
    ) -> None:
        self.registry = registry
        self.decision_provider = decision_provider
        self.runtime = runtime or ToolRuntime(registry)
        self.observation_builder = observation_builder or ObservationBuilder()
        self.policy = policy or AgentPolicy(registry)
        self.stop_gate = stop_gate or StopGate()
        self.result_applier = result_applier
        self.after_iteration = after_iteration
        self.trace_callback = trace_callback
        self.checkpoint_callback = checkpoint_callback
        self.policy_retry_limit = max(0, min(int(policy_retry_limit), 5))
        self.trace = TraceRecorder()
        self._last_stop_rejection_signature: tuple[str, ...] | None = None
        self._consecutive_stop_rejections = 0
        self.runtime.event_callback = self._record_tool_event

    def run(
        self,
        context: Any,
        *,
        max_iterations: int | None = None,
        start_iteration: int = 0,
    ) -> HarnessResult:
        decisions: list[AgentDecision] = []
        tool_results: list[ToolResult] = []
        stop_rejections: list[str] = list(getattr(context, "stop_rejections", []) or [])
        recent_results = _state_tool_results(context)
        self._restore_stop_rejection_state(context)
        iteration = max(0, int(start_iteration))
        turns = 0
        policy_retries = 0

        while max_iterations is None or turns < max_iterations:
            observation = self.observation_builder.build(
                context,
                self.registry,
                iteration=iteration,
                recent_results=recent_results,
                stop_rejections=stop_rejections,
            )
            decision = self.decision_provider(observation, context)
            decisions.append(decision)
            if decision.decision != "stop":
                # A non-stop decision gives the Agent a chance to make
                # progress or repair its plan; it breaks a no-progress stop
                # streak even when the selected call is later rejected.
                self._reset_stop_rejection_state()
            decision_event = self.trace.record_decision(decision, iteration=iteration)
            self._notify_trace(decision_event.model_dump(mode="json"), context)
            _append_context(context, "agent_decision_history", decision.model_dump(mode="json"))
            _set_context(context, "agent_trace", self.trace.model_dump())
            _set_context(context, "runtime_iteration", iteration)

            policy_result = self.policy.validate(decision, context)
            if policy_result.violations and policy_result.tool_calls:
                message = "Policy filtered Agent decision: " + "; ".join(policy_result.violations)
                stop_rejections.extend(policy_result.violations)
                _append_context(context, "stop_rejections", message)
                policy_event = self.trace.record(
                    "policy_filtered",
                    iteration=iteration,
                    status="filtered",
                    payload={"reasons": policy_result.violations},
                )
                self._notify_trace(policy_event.model_dump(mode="json"), context)
            if not policy_result.allowed:
                message = "Policy rejected Agent decision: " + "; ".join(policy_result.violations)
                stop_rejections.extend(policy_result.violations)
                _append_context(context, "stop_rejections", message)
                policy_event = self.trace.record("policy_rejected", iteration=iteration, status="rejected", payload={"reasons": policy_result.violations})
                self._notify_trace(policy_event.model_dump(mode="json"), context)
                if policy_retries < self.policy_retry_limit:
                    policy_retries += 1
                    retry_event = self.trace.record(
                        "policy_retry",
                        iteration=iteration,
                        status="retrying",
                        payload={
                            "retry": policy_retries,
                            "retry_limit": self.policy_retry_limit,
                            "reasons": policy_result.violations,
                        },
                    )
                    self._notify_trace(retry_event.model_dump(mode="json"), context)
                    # Rebuild the observation with the rejection attached,
                    # while keeping this correction inside the same Agent turn.
                    continue
                exhausted_event = self.trace.record(
                    "policy_retry_exhausted",
                    iteration=iteration,
                    status="partial",
                    payload={
                        "retry_limit": self.policy_retry_limit,
                        "reasons": policy_result.violations,
                    },
                )
                self._notify_trace(exhausted_event.model_dump(mode="json"), context)
                policy_retries = 0
                iteration += 1
                turns += 1
                continue

            if decision.decision == "stop":
                stop_result = self.stop_gate.evaluate(decision, context)
                if stop_result.allowed:
                    reason = decision.stop_reason or decision.reason or "Agent requested stop after coverage passed."
                    _set_context(context, "runtime_status", "completed")
                    _set_context(context, "runtime_stop_reason", reason)
                    _set_context(context, "agent_trace", self.trace.model_dump())
                    return HarnessResult(
                        status="completed",
                    iterations=turns + 1,
                        decisions=decisions,
                        tool_results=tool_results,
                        stop_reason=reason,
                        stop_rejections=stop_rejections,
                        trace=self.trace.model_dump(),
                    )
                rejection_signature = tuple(sorted(set(stop_result.reasons)))
                if rejection_signature == self._last_stop_rejection_signature:
                    self._consecutive_stop_rejections += 1
                else:
                    self._last_stop_rejection_signature = rejection_signature
                    self._consecutive_stop_rejections = 1
                stop_rejections.extend(stop_result.reasons)
                _append_context(context, "stop_rejections", *stop_result.reasons)
                stop_event = self.trace.record("stop_rejected", iteration=iteration, status="rejected", payload={"reasons": stop_result.reasons})
                self._notify_trace(stop_event.model_dump(mode="json"), context)
                if self._consecutive_stop_rejections >= 2:
                    reason = (
                        "Agent repeatedly requested stop while the same stop-gate "
                        "blockers remained: " + "; ".join(stop_result.reasons)
                    )
                    _set_context(context, "runtime_status", "partial")
                    _set_context(context, "runtime_stop_reason", reason)
                    repeat_event = self.trace.record(
                        "stop_rejected_repeat",
                        iteration=iteration,
                        status="partial",
                        payload={
                            "reasons": stop_result.reasons,
                            "consecutive_rejections": self._consecutive_stop_rejections,
                        },
                    )
                    self._notify_trace(repeat_event.model_dump(mode="json"), context)
                    _set_context(context, "agent_trace", self.trace.model_dump())
                    return HarnessResult(
                        status="partial",
                        iterations=turns + 1,
                        decisions=decisions,
                        tool_results=tool_results,
                        stop_reason=reason,
                        stop_rejections=stop_rejections,
                        trace=self.trace.model_dump(),
                        terminal=True,
                    )
                iteration += 1
                turns += 1
                policy_retries = 0
                continue

            recent_results = self.runtime.execute_many(
                policy_result.tool_calls,
                context=context,
                options={"iteration": iteration},
            )
            tool_results.extend(recent_results)
            for result in recent_results:
                result_event = self.trace.record_tool_result(result, iteration=iteration)
                self._notify_trace(result_event.model_dump(mode="json"), context)
                _append_context(context, "tool_result_history", result.model_dump(mode="json"))
                _set_context(context, "agent_trace", self.trace.model_dump())
                if self.result_applier is not None:
                    self.result_applier(result, context)
                self._checkpoint_after_tool_result(result, context)
            if self.after_iteration is not None:
                self.after_iteration(iteration, recent_results, context)
            iteration += 1
            turns += 1
            policy_retries = 0

        reason = f"Agent harness iteration budget exhausted: {max_iterations}."
        _set_context(context, "runtime_status", "partial")
        _set_context(context, "runtime_stop_reason", reason)
        _set_context(context, "agent_trace", self.trace.model_dump())
        return HarnessResult(
            status="partial",
            iterations=turns,
            decisions=decisions,
            tool_results=tool_results,
            stop_reason=reason,
            stop_rejections=stop_rejections,
            trace=self.trace.model_dump(),
        )

    def _record_tool_event(self, event: dict[str, Any]) -> None:
        call = event.get("call") or {}
        result = event.get("result") or {}
        trace_event = self.trace.record(
            str(event.get("event_type") or "tool_event"),
            call_id=call.get("call_id"),
            tool_name=call.get("tool_name"),
            status=result.get("status"),
            payload=event,
        )
        self._notify_trace(trace_event.model_dump(mode="json"), None)

    def _notify_trace(self, event: dict[str, Any], context: Any) -> None:
        if self.trace_callback is None:
            return
        try:
            self.trace_callback(event, context)
        except Exception:
            # Observability must never change scientific tool behavior.
            return

    def _checkpoint_after_tool_result(self, result: ToolResult, context: Any) -> None:
        if self.checkpoint_callback is None:
            return
        try:
            self.checkpoint_callback(context)
        except Exception as exc:
            # A persistence problem should be visible without turning a
            # scientifically useful tool result into a task failure.
            event = self.trace.record(
                "checkpoint_failed",
                call_id=result.call_id,
                tool_name=result.tool_name,
                status="failed",
                payload={"error": str(exc)},
            )
            self._notify_trace(event.model_dump(mode="json"), context)

    def _reset_stop_rejection_state(self) -> None:
        self._last_stop_rejection_signature = None
        self._consecutive_stop_rejections = 0

    def _restore_stop_rejection_state(self, context: Any) -> None:
        """Restore one prior stop rejection so resume can stop retrying a no-op."""
        if self._last_stop_rejection_signature is not None:
            return
        for event in reversed(getattr(context, "agent_trace", []) or []):
            if not isinstance(event, dict) or event.get("event_type") != "stop_rejected":
                continue
            payload = event.get("payload") or {}
            reasons = payload.get("reasons") if isinstance(payload, dict) else None
            if not isinstance(reasons, list) or not reasons:
                return
            self._last_stop_rejection_signature = tuple(sorted(set(map(str, reasons))))
            self._consecutive_stop_rejections = 1
            return


def _append_context(context: Any, name: str, *values: Any) -> None:
    target = getattr(context, name, None)
    if isinstance(target, list):
        target.extend(values)


def _set_context(context: Any, name: str, value: Any) -> None:
    if hasattr(context, name):
        setattr(context, name, value)


def _state_tool_results(context: Any) -> list[ToolResult]:
    restored: list[ToolResult] = []
    for item in getattr(context, "tool_result_history", []) or []:
        try:
            restored.append(ToolResult.model_validate(item))
        except Exception:
            continue
    return restored
