from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from scidata_agent.agent.tool_protocol import ToolCall, ToolResult, failed_tool_result, timed_call
from scidata_agent.agent.tool_registry import ToolRegistry


ToolHandler = Callable[[ToolCall, Any, dict[str, Any]], ToolResult]


class ToolRuntime:
    """Execute registered tools with validation, timing and in-run idempotency."""

    def __init__(
        self,
        registry: ToolRegistry,
        handler: ToolHandler | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.registry = registry
        self.handler = handler
        self.event_callback = event_callback
        self.max_retries = max_retries
        self._completed: dict[str, ToolResult] = {}

    def restore_completed(self, results: Iterable[ToolResult | dict[str, Any]]) -> int:
        """Restore successful results from a checkpoint for idempotent resume."""
        restored = 0
        for raw in results:
            try:
                result = raw if isinstance(raw, ToolResult) else ToolResult.model_validate(raw)
            except Exception:
                continue
            if result.status != "completed" or not result.idempotency_key:
                continue
            self._completed[str(result.idempotency_key)] = result.model_copy(deep=True)
            restored += 1
        return restored

    def execute(
        self,
        call: ToolCall,
        *,
        context: Any = None,
        options: dict[str, Any] | None = None,
        handler: ToolHandler | None = None,
    ) -> ToolResult:
        validation_errors = self.registry.validate_call(call)
        if validation_errors:
            result = failed_tool_result(call, "; ".join(validation_errors))
            self._emit("tool_rejected", call, result)
            return result

        key = call.effective_idempotency_key()
        cached = self._completed.get(key)
        if cached is not None:
            result = cached.model_copy(update={"cached": True})
            self._emit("tool_cached", call, result)
            return result

        selected_handler = handler or self.handler
        if selected_handler is None:
            result = failed_tool_result(call, "No handler is registered for the requested tool.")
            self._emit("tool_failed", call, result)
            return result

        self._emit("tool_started", call)
        started = time.perf_counter()
        execution_options = options or {}
        spec = self.registry.require(call.tool_name)
        retry_limit = _retry_limit(spec, execution_options, self.max_retries)
        result = timed_call(selected_handler, call, context, execution_options)
        for retry_count in range(1, retry_limit + 1):
            if result.status != "failed":
                break
            self._emit(
                "tool_retry",
                call,
                result.model_copy(update={"retry_count": retry_count}),
            )
            result = timed_call(selected_handler, call, context, execution_options)
            result.retry_count = retry_count
        if result.elapsed_ms <= 0:
            result.elapsed_ms = (time.perf_counter() - started) * 1000
        result.idempotency_key = key
        if result.status == "completed":
            self._completed[key] = result.model_copy(deep=True)
        self._emit("tool_completed" if result.status == "completed" else "tool_failed", call, result)
        return result

    def _emit(
        self,
        event_type: str,
        call: ToolCall,
        result: ToolResult | None = None,
    ) -> None:
        if self.event_callback is None:
            return
        payload: dict[str, Any] = {"call": call.model_dump(mode="json")}
        if result is not None:
            payload["result"] = result.model_dump(mode="json")
        self.event_callback({"event_type": event_type, **payload})

    def execute_many(
        self,
        calls: list[ToolCall],
        *,
        context: Any = None,
        options: dict[str, Any] | None = None,
    ) -> list[ToolResult]:
        """Execute an independent all-safe batch concurrently and preserve order.

        A mixed batch stays sequential because global workflow tools can depend
        on state changes made by artifact tools in the same Agent decision.
        """
        if len(calls) <= 1 or not all(
            (spec := self.registry.get(call.tool_name)) is not None
            and spec.can_run_parallel
            for call in calls
        ):
            return [self.execute(call, context=context, options=options) for call in calls]

        configured = options.get("max_workers") if options else None
        if configured is None:
            try:
                configured = int(os.getenv("SCIDATA_AGENT_TOOL_WORKERS", "4"))
            except (TypeError, ValueError):
                configured = 4
        workers = max(1, min(int(configured), len(calls)))
        if workers == 1:
            return [self.execute(call, context=context, options=options) for call in calls]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent-tool") as pool:
            futures = [
                pool.submit(self.execute, call, context=context, options=options)
                for call in calls
            ]
            # Preserve the planner's order even though completion order varies.
            return [future.result() for future in futures]


def _retry_limit(spec: Any, options: dict[str, Any], configured: int | None) -> int:
    """Retry network-capable tools without silently repeating local parsing."""
    if not spec.supports_retry:
        return 0
    requested = options.get("max_retries")
    if requested is None:
        requested = configured
    if requested is None:
        try:
            requested = int(os.getenv("SCIDATA_AGENT_TOOL_RETRIES", "1"))
        except (TypeError, ValueError):
            requested = 1
    if "network" not in spec.side_effects and "retry_failed" not in options:
        return 0
    if options.get("retry_failed") is False:
        return 0
    return max(0, min(5, int(requested)))
