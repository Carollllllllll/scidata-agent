from __future__ import annotations

import json
from io import StringIO
from typing import Any

from scidata_agent.agent.mcp_adapter import (
    McpCompatibleServer,
    McpResourceAdapter,
    McpStdioTransport,
    McpToolAdapter,
)
from scidata_agent.agent.schemas import AgentState
from scidata_agent.agent.tool_protocol import ToolCall, ToolResult, ToolSpec
from scidata_agent.agent.tool_registry import ToolRegistry
from scidata_agent.agent.tool_runtime import ToolRuntime


def test_mcp_adapter_maps_tools_and_results(tmp_path) -> None:
    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"value": 42},
        )

    registry = ToolRegistry([ToolSpec(name="collect", description="Collect evidence.")])
    server = McpCompatibleServer(
        McpToolAdapter(registry, ToolRuntime(registry, handler=handler)),
        McpResourceAdapter(),
    )
    state = AgentState(research_question="q", files=[], output_dir=tmp_path)

    assert server.handle("tools/list", {}, state)["tools"][0]["inputSchema"] == {}
    result = server.handle("tools/call", {"name": "collect", "call_id": "c1"}, state)
    assert result["isError"] is False
    assert result["structuredContent"]["data"] == {"value": 42}


def test_mcp_resources_expose_coverage_and_reject_foreign_uri(tmp_path) -> None:
    state = AgentState(research_question="q", files=[], output_dir=tmp_path)
    adapter = McpResourceAdapter()
    resources = adapter.list_resources(state)

    assert {item["name"] for item in resources} == {
        "coverage",
        "source-catalog",
        "evidence",
        "agent-trace",
        "decision-history",
        "tool-history",
    }
    coverage = adapter.read_resource(resources[0]["uri"], state)
    assert coverage["decision"] == "continue"
    try:
        adapter.read_resource("scidata://tasks/other/coverage", state)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign resource URI must be rejected")


def test_mcp_stdio_transport_handles_requests_and_notifications(tmp_path) -> None:
    state = AgentState(research_question="test", files=[], output_dir=tmp_path / "outputs")
    registry = ToolRegistry([ToolSpec(name="ping", description="Ping.")])
    runtime = ToolRuntime(
        registry,
        handler=lambda call, context, options: ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"pong": True},
        ),
    )
    server = McpCompatibleServer(McpToolAdapter(registry, runtime), McpResourceAdapter())
    transport = McpStdioTransport(server, lambda _params: state)
    input_stream = StringIO(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ping"}}\n'
    )
    output_stream = StringIO()

    transport.serve(input_stream, output_stream)
    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]

    assert [response["id"] for response in responses] == [1, 2]
    assert responses[1]["result"]["structuredContent"]["data"] == {"pong": True}
