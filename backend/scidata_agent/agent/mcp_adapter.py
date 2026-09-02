from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any, TextIO
from urllib.parse import quote, unquote

from scidata_agent.agent.schemas import AgentState
from scidata_agent.agent.tool_protocol import ToolCall, ToolResult
from scidata_agent.agent.tool_registry import ToolRegistry
from scidata_agent.agent.tool_runtime import ToolRuntime


AfterCallCallback = Callable[[ToolResult, Any], None]


class McpResourceAdapter:
    """Expose scientific state as MCP-style resources without an MCP dependency."""

    def list_resources(self, state: AgentState) -> list[dict[str, Any]]:
        resources = [
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/coverage",
                "name": "coverage",
                "description": "Deterministic evidence coverage and stop-gate report.",
                "mimeType": "application/json",
            },
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/source-catalog",
                "name": "source-catalog",
                "description": "Normalized sources, artifacts, and processing statuses.",
                "mimeType": "application/json",
            },
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/evidence",
                "name": "evidence",
                "description": "Record-to-source evidence traces.",
                "mimeType": "application/json",
            },
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/agent-trace",
                "name": "agent-trace",
                "description": "Observable Agent decisions, policy events, and tool lifecycle events.",
                "mimeType": "application/json",
            },
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/decision-history",
                "name": "decision-history",
                "description": "LLM decisions and the evidence gaps they were intended to resolve.",
                "mimeType": "application/json",
            },
            {
                "uri": f"scidata://tasks/{quote(state.task_id)}/tool-history",
                "name": "tool-history",
                "description": "Uniform ToolResult envelopes, including failures and idempotency metadata.",
                "mimeType": "application/json",
            },
        ]
        return resources

    def read_resource(self, uri: str, state: AgentState) -> dict[str, Any]:
        prefix = f"scidata://tasks/{quote(state.task_id)}/"
        if not uri.startswith(prefix):
            raise ValueError("Resource URI does not belong to the requested task.")
        resource_name = unquote(uri[len(prefix):])
        if resource_name == "coverage":
            return state.coverage_report.model_dump(mode="json")
        if resource_name == "source-catalog":
            return {"items": [item.model_dump(mode="json") for item in state.source_catalog]}
        if resource_name == "evidence":
            return {"items": [item.model_dump(mode="json") for item in state.evidence_traces]}
        if resource_name == "agent-trace":
            return {"items": list(state.agent_trace)}
        if resource_name == "decision-history":
            return {"items": list(state.agent_decision_history)}
        if resource_name == "tool-history":
            return {"items": list(state.tool_result_history)}
        raise KeyError(f"Unknown resource: {resource_name!r}")


class McpToolAdapter:
    """MCP-shaped facade over the dependency-free internal tool protocol."""

    def __init__(
        self,
        registry: ToolRegistry,
        runtime: ToolRuntime,
        *,
        after_call: AfterCallCallback | None = None,
    ) -> None:
        self.registry = registry
        self.runtime = runtime
        self.after_call = after_call

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "inputSchema": spec["input_schema"],
            }
            for spec in self.registry.describe()
        ]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: Any = None,
        call_id: str | None = None,
    ) -> dict[str, Any]:
        call = ToolCall(
            call_id=call_id or f"mcp_{name}",
            tool_name=name,
            arguments=dict(arguments or {}),
            purpose="MCP tool invocation",
        )
        result = self.runtime.execute(call, context=context)
        if self.after_call is not None:
            try:
                self.after_call(result, context)
            except Exception as exc:
                result.warnings.append(f"MCP state persistence failed: {exc}")
        return self._to_mcp_result(result)

    @staticmethod
    def _to_mcp_result(result: ToolResult) -> dict[str, Any]:
        return {
            "isError": result.status == "failed",
            "structuredContent": result.model_dump(mode="json"),
            "content": [
                {
                    "type": "text",
                    "text": (
                        "; ".join(result.errors)
                        if result.errors
                        else f"Tool {result.tool_name} returned {result.status}."
                    ),
                }
            ],
        }


class McpCompatibleServer:
    """Small request router suitable for an MCP transport adapter."""

    def __init__(self, tool_adapter: McpToolAdapter, resource_adapter: McpResourceAdapter) -> None:
        self.tool_adapter = tool_adapter
        self.resource_adapter = resource_adapter

    def capabilities(self) -> dict[str, Any]:
        return {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False}}

    def handle(self, method: str, params: dict[str, Any] | None, state: AgentState) -> dict[str, Any]:
        params = dict(params or {})
        if method in {"tools/list", "list_tools"}:
            return {"tools": self.tool_adapter.list_tools()}
        if method in {"tools/call", "call_tool"}:
            return self.tool_adapter.call_tool(
                str(params.get("name") or ""),
                params.get("arguments") if isinstance(params.get("arguments"), dict) else {},
                context=state,
                call_id=str(params.get("call_id") or "") or None,
            )
        if method in {"resources/list", "list_resources"}:
            return {"resources": self.resource_adapter.list_resources(state)}
        if method in {"resources/read", "read_resource"}:
            uri = str(params.get("uri") or "")
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(
                            self.resource_adapter.read_resource(uri, state),
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        if method == "initialize":
            return {"protocolVersion": "2025-06-18", "capabilities": self.capabilities()}
        raise ValueError(f"Unsupported MCP method: {method!r}")


class McpStdioTransport:
    """Dependency-free JSON-RPC stdio transport for the MCP-shaped server."""

    def __init__(
        self,
        server: McpCompatibleServer,
        state_provider: Callable[[dict[str, Any]], AgentState],
    ) -> None:
        self.server = server
        self.state_provider = state_provider

    def serve(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        input_stream = input_stream or sys.stdin
        output_stream = output_stream or sys.stdout
        for line in input_stream:
            if not line.strip():
                continue
            response = self.handle_line(line)
            if response is None:
                continue
            output_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            output_stream.flush()

    def handle_line(self, line: str) -> dict[str, Any] | None:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("JSON-RPC request must be an object.")
            request_id = request.get("id")
            method = str(request.get("method") or "")
            params = request.get("params")
            if not isinstance(params, dict):
                params = {}
            # MCP notifications intentionally do not receive a response.
            if request_id is None:
                return None
            state = self.state_provider(params)
            result = self.server.handle(method, params, state)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(exc)},
            }
