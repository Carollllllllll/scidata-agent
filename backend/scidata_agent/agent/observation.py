from __future__ import annotations

from collections import Counter
import os
from typing import Any

from pydantic import BaseModel, Field

from scidata_agent.agent.schemas import AgentState
from scidata_agent.agent.tool_protocol import ToolResult
from scidata_agent.agent.tool_registry import ToolRegistry


class AgentObservation(BaseModel):
    """Compact state view presented to the decision model."""

    iteration: int = 0
    task: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    connectors: list[dict[str, Any]] = Field(default_factory=list)
    recent_results: list[dict[str, Any]] = Field(default_factory=list)
    available_tools: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    stop_rejections: list[str] = Field(default_factory=list)


class ObservationBuilder:
    """Build bounded observations without serializing the complete AgentState."""

    def build(
        self,
        state: AgentState,
        registry: ToolRegistry,
        *,
        iteration: int = 0,
        recent_results: list[ToolResult] | None = None,
        stop_rejections: list[str] | None = None,
    ) -> AgentObservation:
        coverage = state.coverage_report
        discovery = state.source_discovery_plan
        artifacts = [
            artifact
            for entry in state.source_catalog
            for artifact in entry.artifacts
        ]
        source_limit = _observation_limit("SCIDATA_AGENT_OBSERVATION_MAX_SOURCES", 40)
        artifact_limit = _observation_limit("SCIDATA_AGENT_OBSERVATION_MAX_ARTIFACTS", 80)
        selected_sources = _rank_sources(state.source_catalog)[:source_limit]
        selected_artifacts = _rank_artifacts(artifacts)[:artifact_limit]
        source_statuses = Counter(entry.status for entry in state.source_catalog)
        artifact_statuses = Counter(artifact.status for artifact in artifacts)
        failures = [
            str(artifact.failure_reason or f"{artifact.artifact_id}: failed")
            for artifact in artifacts
            if artifact.status == "failed"
        ]
        failures.extend(
            str(item.get("error"))
            for item in getattr(state, "tool_result_history", [])
            if isinstance(item, dict) and item.get("status") == "failed" and item.get("error")
        )
        failures.extend(
            _connector_failure_summary(item)
            for item in getattr(state, "connector_status", [])
            if isinstance(item, dict) and item.get("status") in {"failed", "error"}
        )
        observed_results = _recent_tool_results(state, recent_results)
        connector_rows = _connector_summary(getattr(state, "connector_status", []))
        return AgentObservation(
            iteration=iteration,
            task={
                "research_question": state.research_question,
                "target_fields": list(state.task_plan.target_fields) if state.task_plan else [],
                "required_evidence_types": list(
                    state.dynamic_extraction_plan.source_requirements
                    if state.dynamic_extraction_plan
                    else []
                ),
            },
            progress={
                "coverage_decision": coverage.decision,
                "coverage_score": coverage.coverage_score,
                "missing_requirements": list(coverage.missing_requirements),
                "coverage_gaps": len(coverage.gaps),
                "open_quality_issues": state.quality_report.issue_count,
                "open_conflicts": state.quality_report.conflict_count,
                "materialized_files": len(state.files),
                "text_blocks": len(state.parsed_sources.text_blocks),
                "section_blocks": len(state.parsed_sources.section_blocks),
                "tables": len(state.parsed_sources.tables),
                "figures": len(state.chart_extractions),
                "dynamic_records": len(state.dynamic_records),
                "candidate_records": len(state.candidate_records),
                "final_records": len(state.final_records),
                "search_more_count": int(getattr(state, "runtime_search_more_count", 0)),
                "search_more_limit": int(getattr(state, "runtime_search_more_limit", 2)),
                "group_initial_searches": list(state.runtime_group_initial_searches),
                "group_search_more_counts": dict(state.runtime_group_search_more_counts),
                "completed_content_stages": [
                    str(item.get("tool_name"))
                    for item in getattr(state, "tool_result_history", [])
                    if isinstance(item, dict)
                    and item.get("status") == "completed"
                    and str(item.get("tool_name")) in {
                        "parse_source_content",
                        "extract_figures",
                        "interpret_sections",
                        "extract_dynamic_records",
                        "extract_records",
                        "normalize_records",
                        "track_provenance",
                        "parse_content",
                    }
                ],
            },
            sources={
                "candidate_count": len(discovery.candidate_sources) if discovery else 0,
                "catalog_count": len(state.source_catalog),
                "status_counts": dict(source_statuses),
                "items": [
                    {
                        "source_id": source.source_id,
                        "title": source.title,
                        "provider": source.provider,
                        "status": source.status,
                        "relevance_score": source.relevance_score,
                        "artifact_count": len(source.artifacts),
                    }
                    for source in selected_sources
                ],
            },
            artifacts={
                "total": len(artifacts),
                "status_counts": dict(artifact_statuses),
                "unprocessed_relevant": len(coverage.unprocessed_relevant_artifacts),
                "items": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "source_id": artifact.source_id,
                        "name": artifact.name,
                        "artifact_type": artifact.artifact_type,
                        "status": artifact.status,
                        "relevance_score": artifact.relevance_score,
                        "url": artifact.url,
                        "local_path": artifact.local_path,
                        "failure_reason": artifact.failure_reason,
                    }
                    for artifact in selected_artifacts
                ],
            },
            recent_results=[
                result.model_dump(mode="json")
                for result in observed_results[-12:]
            ],
            connectors=connector_rows,
            available_tools=[
                tool
                for tool in registry.describe()
                if not (
                    tool.get("name") == "search_more"
                    and not _search_more_available(state)
                )
            ],
            failures=failures[-12:],
            stop_rejections=list(stop_rejections or [])[-8:],
        )


def _observation_limit(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _search_more_available(state: AgentState) -> bool:
    limit = int(getattr(state, "runtime_search_more_limit", 2))
    groups = state.coverage_report.field_groups
    if groups:
        return any(
            group.status == "pending"
            or (
                group.status == "insufficient"
                and group.search_more_count < limit
            )
            for group in groups
        )
    return int(getattr(state, "runtime_search_more_count", 0)) < limit


def _rank_sources(sources: list[Any]) -> list[Any]:
    """Keep failed and high-value sources visible while bounding prompt size."""
    return sorted(
        sources,
        key=lambda source: (
            source.status != "failed",
            -(float(source.relevance_score or 0.0)),
            source.status not in {"selected", "downloaded", "parsed"},
            str(source.source_id),
        ),
    )


def _rank_artifacts(artifacts: list[Any]) -> list[Any]:
    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.status != "failed",
            -(float(artifact.relevance_score or 0.0)),
            artifact.status not in {"discovered", "selected", "planned"},
            str(artifact.artifact_id),
        ),
    )


def _connector_failure_summary(item: dict[str, Any]) -> str:
    connector = item.get("connector") or item.get("connector_name") or "unknown connector"
    request = item.get("query") or item.get("request") or ""
    error = item.get("error") or item.get("message") or "connector request failed"
    suffix = f" query={request!r}" if request else ""
    return f"Connector {connector!r} failed{suffix}: {error}"


def _connector_summary(status_items: list[Any]) -> list[dict[str, Any]]:
    """Collapse per-connector status records into a compact usable/not-usable view.

    The decision model needs to know which connectors are currently usable so it
    can avoid selecting download sources that originate from a failed connector.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for item in status_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("connector") or item.get("connector_name") or "unknown")
        status = str(item.get("status") or "")
        entry = by_name.setdefault(
            name, {"connector": name, "statuses": [], "error": ""}
        )
        entry["statuses"].append(status)
        if status in {"failed", "error"} and not entry["error"]:
            entry["error"] = str(item.get("error") or item.get("message") or "")[:200]
    rows: list[dict[str, Any]] = []
    for name in sorted(by_name):
        entry = by_name[name]
        usable = any(status in {"completed", "partial"} for status in entry["statuses"])
        rows.append(
            {
                "connector": name,
                "usable": usable,
                "statuses": entry["statuses"][:4],
                "error": entry["error"],
            }
        )
    return rows


def _recent_tool_results(
    state: AgentState,
    recent_results: list[ToolResult] | None,
) -> list[ToolResult]:
    if recent_results is not None:
        return recent_results
    restored: list[ToolResult] = []
    for item in getattr(state, "tool_result_history", []):
        try:
            restored.append(ToolResult.model_validate(item))
        except Exception:
            continue
    return restored
