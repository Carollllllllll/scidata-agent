from __future__ import annotations

import json

from scidata_agent.agent.checkpoint import AgentCheckpointStore, build_run_fingerprint
from scidata_agent.agent.schemas import AgentState, SourceArtifact, SourceCatalogEntry
from scidata_agent.mcp_server import McpTaskServer


def _write_checkpoint(tmp_path, state: AgentState) -> None:
    fingerprint = build_run_fingerprint(state.research_question, [], {"test": True})
    AgentCheckpointStore(tmp_path).save(
        state,
        fingerprint=fingerprint,
        completed_steps=set(),
    )


def test_checkpoint_backed_mcp_server_executes_and_persists_local_tool(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    task_dir = tmp_path / "task"
    state = AgentState(
        task_id="mcp-test",
        research_question="Read the metric table.",
        files=[],
        output_dir=tmp_path,
        source_catalog=[
            SourceCatalogEntry(
                source_id="source-1",
                title="Local metrics",
                artifacts=[
                    SourceArtifact(
                        source_id="source-1",
                        name="metrics.csv",
                        artifact_type="csv",
                        local_path=str(csv_path),
                    )
                ],
            )
        ],
    )
    _write_checkpoint(task_dir, state)

    server = McpTaskServer(task_dir)
    listed = server.server.handle("tools/list", {}, server.state)
    assert any(tool["name"] == "parse_csv" for tool in listed["tools"])

    response = server.server.handle(
        "tools/call",
        {"name": "parse_csv", "arguments": {"artifact_id": "" + state.source_catalog[0].artifacts[0].artifact_id}},
        server.state,
    )
    assert response["isError"] is False
    assert response["structuredContent"]["status"] == "completed"
    assert len(server.state.parsed_sources.tables) == 1

    persisted = json.loads((task_dir / "agent_checkpoint.json").read_text(encoding="utf-8"))
    assert persisted["state"]["tool_result_history"][0]["tool_name"] == "parse_csv"

    restored_server = McpTaskServer(task_dir)
    cached = restored_server.server.handle(
        "tools/call",
        {"name": "parse_csv", "arguments": {"artifact_id": state.source_catalog[0].artifacts[0].artifact_id}},
        restored_server.state,
    )
    assert cached["structuredContent"]["cached"] is True


def test_checkpoint_backed_mcp_server_exposes_scientific_resources(tmp_path) -> None:
    task_dir = tmp_path / "task"
    state = AgentState(
        task_id="mcp-resource-test",
        research_question="Inspect evidence.",
        files=[],
        output_dir=tmp_path,
    )
    _write_checkpoint(task_dir, state)
    server = McpTaskServer(task_dir)

    resources = server.server.handle("resources/list", {}, server.state)["resources"]
    names = {item["name"] for item in resources}
    assert {
        "coverage",
        "source-catalog",
        "evidence",
        "agent-trace",
        "decision-history",
        "tool-history",
    } <= names
