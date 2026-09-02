from __future__ import annotations

from typing import Any

from scidata_agent.agent.tool_protocol import ToolCall, ToolResult, ToolSpec
from scidata_agent.agent.tool_registry import ToolRegistry, build_artifact_tool_registry
from scidata_agent.agent.tool_runtime import ToolRuntime


def test_tool_registry_describes_json_schema_and_validates_required_arguments() -> None:
    registry = ToolRegistry(
        [
            ToolSpec(
                name="read_evidence",
                description="Read one evidence item.",
                input_schema={
                    "type": "object",
                    "properties": {"evidence_id": {"type": "string"}},
                    "required": ["evidence_id"],
                },
            )
        ]
    )

    descriptions = registry.describe()
    assert descriptions[0]["name"] == "read_evidence"
    assert descriptions[0]["input_schema"]["required"] == ["evidence_id"]
    assert registry.validate_call(ToolCall(call_id="call-1", tool_name="read_evidence")) == [
        "Missing required tool argument: evidence_id"
    ]
    assert registry.validate_call(
        ToolCall(call_id="call-2", tool_name="read_evidence", arguments={"evidence_id": "ev-1"})
    ) == []


def test_tool_runtime_returns_uniform_result_and_reuses_completed_call() -> None:
    calls: list[str] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"value": context["value"]},
        )

    registry = ToolRegistry(
        [ToolSpec(name="read_value", description="Read a value.", input_schema={"type": "object"})]
    )
    runtime = ToolRuntime(registry, handler=handler)
    first = runtime.execute(
        ToolCall(call_id="call-1", tool_name="read_value", arguments={"id": "same"}),
        context={"value": 42},
    )
    second = runtime.execute(
        ToolCall(call_id="call-2", tool_name="read_value", arguments={"id": "same"}),
        context={"value": 99},
    )

    assert first.status == "completed"
    assert first.data == {"value": 42}
    assert first.idempotency_key
    assert second.status == "completed"
    assert second.cached is True
    assert second.data == {"value": 42}
    assert calls == ["call-1"]


def test_tool_runtime_rejects_unknown_tool_and_does_not_invoke_handler() -> None:
    invoked = False

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        nonlocal invoked
        invoked = True
        raise AssertionError("invalid call must be rejected before the handler")

    runtime = ToolRuntime(
        ToolRegistry([ToolSpec(name="known", description="Known tool.")]),
        handler=handler,
    )

    result = runtime.execute(ToolCall(call_id="call-1", tool_name="unknown"))

    assert result.status == "failed"
    assert result.errors == ["Unknown tool: 'unknown'"]
    assert invoked is False


def test_tool_runtime_does_not_cache_failure_and_allows_retry() -> None:
    attempts = 0

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary provider outage")
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    runtime = ToolRuntime(
        ToolRegistry([ToolSpec(name="retryable", description="Retryable tool.")]),
        handler=handler,
    )
    call = ToolCall(call_id="call-1", tool_name="retryable", arguments={"item": "same"})

    first = runtime.execute(call)
    second = runtime.execute(call.model_copy(update={"call_id": "call-2"}))

    assert first.status == "failed"
    assert second.status == "completed"
    assert second.cached is False
    assert attempts == 2


def test_tool_runtime_restores_completed_results_for_resume() -> None:
    registry = ToolRegistry([ToolSpec(name="read_value", description="Read a value.")])
    calls: list[str] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"value": 7},
        )

    first_runtime = ToolRuntime(registry, handler=handler)
    first = first_runtime.execute(
        ToolCall(call_id="call-1", tool_name="read_value", arguments={"id": "same"})
    )
    resumed_runtime = ToolRuntime(registry, handler=handler)
    assert resumed_runtime.restore_completed([first.model_dump(mode="json")]) == 1

    restored = resumed_runtime.execute(
        ToolCall(call_id="call-2", tool_name="read_value", arguments={"id": "same"})
    )

    assert restored.cached is True
    assert restored.data == {"value": 7}
    assert calls == ["call-1"]


def test_artifact_registry_adapts_existing_actions_to_tool_specs() -> None:
    registry = build_artifact_tool_registry()
    names = {spec.name for spec in registry.list()}

    assert {"search_more", "download_artifact", "parse_table", "parse_figure", "stop"} <= names
    assert registry.require("parse_table").requires_artifact is True
    assert registry.require("search_more").global_action is True
    assert registry.require("parse_table").input_schema["required"] == ["artifact_id"]


def test_runtime_retries_network_tool_and_records_retry_count() -> None:
    attempts = 0
    events: list[dict[str, Any]] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="failed", errors=["503"])
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([ToolSpec(name="search", description="Search.", side_effects={"network"})])
    runtime = ToolRuntime(registry, handler=handler, event_callback=events.append, max_retries=1)
    result = runtime.execute(ToolCall(call_id="c1", tool_name="search"))

    assert result.status == "completed"
    assert result.retry_count == 1
    assert attempts == 2
    assert any(event["event_type"] == "tool_retry" for event in events)
