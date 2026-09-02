from __future__ import annotations

from typing import Any

from scidata_agent.agent.decision import AgentDecision, PolicyDecision
from scidata_agent.agent.schemas import AgentState
from scidata_agent.agent.tool_protocol import ToolCall
from scidata_agent.agent.tool_registry import ToolRegistry


# These actions consume or validate task-specific evidence.  They must not
# run before the LLM has materialized the contract that defines what evidence
# is relevant and how it should be represented.
_TASK_CONTRACT_ACTIONS = frozenset({
    "discover_sources",
    "plan_multi_source_search",
    "search_sources",
    "search_more",
    "read_metadata",
    "download_artifact",
    "parse_pdf_text",
    "parse_pdf_sections",
    "parse_table",
    "parse_figure",
    "parse_html",
    "parse_csv",
    "read_readme",
    "read_file_manifest",
    "validate_evidence",
    "select_sources",
    "triage_sources",
    "ingest_sources",
    "ingest_arxiv_pdfs",
    "parse_content",
    "parse_source_content",
    "extract_figures",
    "interpret_sections",
    "extract_dynamic_records",
    "extract_records",
    "normalize_records",
    "track_provenance",
    "validate_quality",
})


class AgentPolicy:
    """Deterministic guardrails around model-authored tool decisions."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def validate(self, decision: AgentDecision, state: AgentState) -> PolicyDecision:
        violations: list[str] = []
        if decision.decision == "stop" and decision.tool_calls:
            violations.append("A stop decision must not contain tool calls.")
            return PolicyDecision(allowed=False, tool_calls=[], violations=violations)
        if decision.decision == "continue" and not decision.tool_calls:
            violations.append("A continue decision must contain at least one tool call.")
            return PolicyDecision(allowed=False, tool_calls=[], violations=violations)
        seen: set[str] = set()
        completed_keys = {
            str(item.get("idempotency_key"))
            for item in getattr(state, "tool_result_history", [])
            if isinstance(item, dict)
            and item.get("status") == "completed"
            and item.get("idempotency_key")
        }
        # A single model decision may contain an ordered workflow such as
        # plan_task -> plan_dynamic_schema -> discover_sources. Validate each
        # call against the state that would exist immediately before it, while
        # keeping the real state immutable until the runtime executes it.
        readiness = {
            "task_plan": state.task_plan is not None,
            "dynamic_schema": state.dynamic_extraction_plan is not None,
            "source_discovery": state.source_discovery_plan is not None,
            "search_plan": state.multi_source_search_plan is not None,
            "search_executed": _has_search_attempt(state),
            "content": bool(
                state.files
                or state.parsed_sources.text_blocks
                or state.parsed_sources.tables
            ),
            "candidate_records": bool(state.candidate_records),
        }
        valid_calls: list[ToolCall] = []
        for call in decision.tool_calls:
            call_violations = self.registry.validate_call(call)
            call_violations.extend(
                _workflow_prerequisite_violations(call, state, readiness=readiness)
            )
            key = call.effective_idempotency_key()
            if key in seen:
                call_violations.append(f"Duplicate tool call in one decision: {call.tool_name!r}.")
            seen.add(key)
            if key in completed_keys:
                call_violations.append(f"Tool call already completed: {call.tool_name!r}.")
            artifact_id = call.arguments.get("artifact_id")
            if artifact_id and not _artifact_exists(state, str(artifact_id)):
                call_violations.append(f"Artifact not found: {artifact_id!r}.")
            if call_violations:
                violations.extend(call_violations)
                continue
            valid_calls.append(call)
            _project_tool_effect(call.tool_name, readiness)
        # Keep valid calls from a mixed decision while filtering only the
        # repeated or otherwise invalid calls. Reject an all-invalid batch so
        # the next observation can explain the problem to the model.
        allowed = bool(valid_calls) if decision.tool_calls else not violations
        return PolicyDecision(
            allowed=allowed,
            tool_calls=valid_calls,
            violations=list(dict.fromkeys(violations)),
        )


def _artifact_exists(state: AgentState, artifact_id: str) -> bool:
    return any(
        artifact.artifact_id == artifact_id
        for entry in state.source_catalog
        for artifact in entry.artifacts
    )


def _workflow_prerequisite_violations(
    call: ToolCall,
    state: AgentState,
    *,
    readiness: dict[str, bool] | None = None,
) -> list[str]:
    """Reject content-stage calls whose required state has not been materialized."""
    action = call.tool_name
    violations: list[str] = []
    effective = readiness or {
        "task_plan": state.task_plan is not None,
        "dynamic_schema": state.dynamic_extraction_plan is not None,
        "source_discovery": state.source_discovery_plan is not None,
        "search_plan": state.multi_source_search_plan is not None,
        "search_executed": _has_search_attempt(state),
        "content": bool(
            state.files
            or state.parsed_sources.text_blocks
            or state.parsed_sources.tables
        ),
        "candidate_records": bool(state.candidate_records),
    }
    if action in _TASK_CONTRACT_ACTIONS:
        if not effective["task_plan"]:
            violations.append(f"{action} requires a completed task plan first.")
        if not effective["dynamic_schema"]:
            violations.append(
                f"{action} requires a completed dynamic extraction schema first."
            )
    if action == "plan_dynamic_schema" and not effective["task_plan"]:
        violations.append("plan_dynamic_schema requires a completed task plan.")
    if action in {"plan_multi_source_search", "search_sources"}:
        if not effective["source_discovery"]:
            violations.append(f"{action} requires completed source discovery.")
        elif action == "search_sources" and not effective["search_plan"]:
            violations.append("search_sources requires a completed multi-source search plan.")

    if state.runtime_requires_source_discovery and action in {
        "select_sources",
        "triage_sources",
        "ingest_sources",
        "ingest_arxiv_pdfs",
    } and not effective["search_executed"]:
        violations.append(
            f"{action} requires a successful or partial multi-source search attempt first."
        )

    if state.runtime_requires_source_discovery and action == "download_artifact":
        artifact_id = str(call.arguments.get("artifact_id") or "")
        artifact = _find_artifact(state, artifact_id)
        if artifact is not None and not artifact.local_path and not effective["search_executed"]:
            violations.append(
                "download_artifact for a remote catalog artifact requires a successful "
                "or partial multi-source search attempt first."
            )

    if action in {
        "interpret_sections",
        "extract_dynamic_records",
        "extract_records",
    } and not effective["content"]:
        violations.append(f"{action} requires parsed text or table evidence first.")
    if action == "extract_dynamic_records":
        if not effective["dynamic_schema"]:
            violations.append("extract_dynamic_records requires a completed dynamic extraction schema.")
    if action == "extract_records" and not effective["task_plan"]:
        violations.append("extract_records requires a completed task plan.")
    if action == "normalize_records" and not effective["candidate_records"]:
        violations.append("normalize_records requires extracted candidate records first.")
    return violations


def _project_tool_effect(action: str, readiness: dict[str, bool]) -> None:
    """Project only known state-producing tools for same-decision validation."""
    if action == "plan_task":
        readiness["task_plan"] = True
    elif action == "plan_dynamic_schema":
        readiness["dynamic_schema"] = True
    elif action == "discover_sources":
        readiness["source_discovery"] = True
    elif action == "plan_multi_source_search":
        readiness["search_plan"] = True
    elif action in {"search_sources", "search_more"}:
        readiness["search_executed"] = True
    elif action in {
        "parse_pdf_text",
        "parse_pdf_sections",
        "parse_table",
        "parse_csv",
        "parse_source_content",
        "interpret_sections",
        "extract_figures",
        "extract_dynamic_records",
        "extract_records",
    }:
        readiness["content"] = True
    if action in {"extract_dynamic_records", "extract_records"}:
        readiness["candidate_records"] = True


def _has_search_attempt(state: AgentState) -> bool:
    """Treat partial provider success as usable progress, but not failures."""
    return any(
        isinstance(item, dict)
        and item.get("tool_name") in {"search_sources", "search_more"}
        and item.get("status") in {"completed", "partial"}
        for item in getattr(state, "tool_result_history", [])
    )


def _find_artifact(state: AgentState, artifact_id: str) -> Any | None:
    if not artifact_id:
        return None
    for entry in state.source_catalog:
        for artifact in entry.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
    return None
