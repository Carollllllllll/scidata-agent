from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scidata_agent.agent.action_executor import ArtifactActionExecutor
from scidata_agent.agent.checkpoint import AgentCheckpointStore, CHECKPOINT_VERSION
from scidata_agent.agent.mcp_adapter import (
    McpCompatibleServer,
    McpResourceAdapter,
    McpStdioTransport,
    McpToolAdapter,
)
from scidata_agent.agent.schemas import AgentState
from scidata_agent.agent.tool_registry import ToolRegistry
from scidata_agent.agent.tool_runtime import ToolRuntime


# This standalone process is intentionally limited to tools that do not need
# an in-process LLM decision provider. The main Agent exposes the full set.
STANDALONE_TOOL_NAMES = frozenset(
    {
        "read_metadata",
        "download_artifact",
        "parse_pdf_text",
        "parse_table",
        "parse_csv",
        "parse_html",
        "read_readme",
        "read_file_manifest",
    }
)


class McpTaskServer:
    """MCP stdio facade for one persisted scientific task."""

    def __init__(self, task_dir: str | Path) -> None:
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.state, self.fingerprint, self.completed_steps = _load_checkpoint(self.task_dir)
        self.executor = ArtifactActionExecutor()
        self.registry = ToolRegistry(
            spec
            for spec in self.executor.tool_registry.list()
            if spec.name in STANDALONE_TOOL_NAMES
        )
        self.runtime = ToolRuntime(self.registry, handler=self.executor._handle_tool_call)
        self.runtime.restore_completed(self.state.tool_result_history)
        self.tool_adapter = McpToolAdapter(
            self.registry,
            self.runtime,
            after_call=self._persist_after_call,
        )
        self.server = McpCompatibleServer(self.tool_adapter, McpResourceAdapter())

    def state_provider(self, _params: dict[str, Any]) -> AgentState:
        return self.state

    def serve(self) -> None:
        McpStdioTransport(self.server, self.state_provider).serve()

    def _persist_after_call(self, result, context: Any) -> None:
        state = context if isinstance(context, AgentState) else self.state
        self.state = state
        result_payload = result.model_dump(mode="json")
        already_recorded = any(
            isinstance(item, dict)
            and item.get("call_id") == result.call_id
            and item.get("idempotency_key") == result.idempotency_key
            and item.get("status") == result.status
            for item in state.tool_result_history
        )
        if not already_recorded:
            state.tool_result_history.append(result_payload)
        if result.status == "completed":
            self.runtime.restore_completed([result])
        AgentCheckpointStore(self.task_dir).save(
            state,
            fingerprint=self.fingerprint,
            completed_steps=self.completed_steps,
        )


def _load_checkpoint(task_dir: Path) -> tuple[AgentState, str, set[str]]:
    checkpoint_path = task_dir / "agent_checkpoint.json"
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read MCP task checkpoint: {checkpoint_path}") from exc
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version: {payload.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}."
        )
    fingerprint = str(payload.get("fingerprint") or "")
    state_payload = payload.get("state")
    completed = payload.get("completed_steps")
    if not fingerprint or not isinstance(state_payload, dict) or not isinstance(completed, list):
        raise RuntimeError("MCP task checkpoint is missing fingerprint, state, or completed_steps.")
    try:
        state = AgentState.model_validate(state_payload)
    except Exception as exc:
        raise RuntimeError("MCP task checkpoint contains an invalid AgentState.") from exc
    return state, fingerprint, {str(item) for item in completed if str(item).strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve one SciData Agent checkpoint through MCP-compatible stdio JSON-RPC."
    )
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Task directory containing agent_checkpoint.json, normally outputs/<task_id>.",
    )
    args = parser.parse_args(argv)
    McpTaskServer(args.task_dir).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
