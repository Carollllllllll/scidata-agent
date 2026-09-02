# SciData Agent Tool Protocol

The backend uses a small, dependency-free tool contract as the first step
toward MCP compatibility. Existing Python tools remain in-process; the
contract makes their discovery, invocation, and results independent from the
implementation behind them.

## Contract

`ToolSpec` describes a tool and its JSON-compatible input/output schemas.
`ToolCall` is a model-authored invocation with an explicit `call_id`, reason,
expected evidence, and an idempotency key. `ToolResult` is the single result
envelope used by the runtime:

```text
ToolSpec -> ToolCall -> ToolRuntime -> ToolResult
```

Every result includes a status, structured data, artifact/evidence references,
warnings, errors, elapsed time, and idempotency metadata. A failed call is not
cached, so a later retry or provider switch can execute it again.

## Compatibility mapping

The current artifact action system is adapted without changing its public
behavior:

```text
ActionCapability      -> ToolSpec
ArtifactAction        -> ToolCall
ArtifactActionResult  -> ToolResult
ArtifactActionExecutor -> ToolRuntime bridge
```

`ArtifactActionExecutor.execute_action()` remains available for existing
callers. `execute_plan()` now routes calls through `ToolRuntime` and converts
the result back to `ArtifactActionResult` for API and checkpoint compatibility.
In dynamic runtime mode, the same registry also exposes global workflow tools
for source selection, triage, ingestion, content parsing, and quality
validation. Their handlers call the existing scientific pipeline functions;
they do not replace TATR, VL, PDF parsing, or quality logic.

## MCP relationship

The fields intentionally correspond to the information an MCP client needs
to discover and invoke a tool. The dependency-free `mcp_adapter.py` module
now exposes the same registry as MCP-shaped tools and exposes coverage,
source catalog, and evidence traces as resources. `McpCompatibleServer` is the
transport-neutral request router and `McpStdioTransport` provides a
dependency-free newline-delimited JSON-RPC stdio entry point. The existing
FastAPI API remains the HTTP task interface; no second HTTP MCP dependency is
needed. Keeping both paths on the same contract prevents a network protocol
from changing scientific extraction behavior.

## Standalone task server

The checkpoint-backed stdio entry point can be started for one completed or
partial task:

```powershell
python -m scidata_agent.mcp_server --task-dir "..\outputs\<task_id>"
```

It exposes local artifact operations and the `coverage`, `source-catalog`,
`evidence`, `agent-trace`, `decision-history`, and `tool-history` resources.
Each tool call atomically updates the task checkpoint. LLM-directed workflow
operations remain owned by the main Agent Harness so the standalone adapter
does not pretend to have an unavailable decision provider.

## Runtime rules

- Unknown tools and missing required arguments are rejected before execution.
- Completed calls are cached by `tool_name` and canonical arguments.
- Failed and skipped calls are not cached as successful work.
- Network-capable tools retry failed results up to
  `SCIDATA_AGENT_TOOL_RETRIES` (default `1`, maximum `5`); local parsing tools
  are not retried unless the caller explicitly supplies `retry_failed`.
- Legacy action validation remains in the bridge during migration.
- An all-safe batch whose tools declare `can_run_parallel` is executed by the
  runtime worker pool and returned in planner order. Mixed batches remain
  sequential because global workflow actions may depend on shared state
  mutations from artifact tools.
- Successful tool results and decision history are persisted in AgentState
  checkpoints, and runtime calls can restore completed results after restart.
- Completed tasks additionally export `agent_trace.json`,
  `decision_history.json`, and `tool_history.json` under the task output
  directory and expose their paths through `export_files`.
