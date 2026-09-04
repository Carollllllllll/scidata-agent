from __future__ import annotations

from scidata_agent.agent.decision import AgentDecision, StopDecision
from scidata_agent.agent.schemas import AgentState


class StopGate:
    """Allow termination only when deterministic coverage checks are satisfied."""

    def evaluate(self, decision: AgentDecision, state: AgentState) -> StopDecision:
        if decision.decision != "stop":
            return StopDecision(allowed=False, reasons=["The decision did not request stop."])
        coverage = state.coverage_report
        reasons: list[str] = []
        field_group_work_complete = _field_group_work_complete(state)
        if coverage.decision != "allow_stop" and not field_group_work_complete:
            reasons.append(
                f"Coverage gate is {coverage.decision!r}; required evidence is not complete."
            )
        if coverage.missing_requirements and not field_group_work_complete:
            reasons.append(
                "Missing requirements: " + ", ".join(coverage.missing_requirements[:12])
            )
        # Keep this definition aligned with build_coverage_report(). An
        # unavailable gap remains visible for audit, but another Agent turn
        # cannot repair it and must not create an endless stop-rejection loop.
        blocking_gaps = [
            gap for gap in coverage.gaps
            if gap.priority != "low" and gap.status != "unavailable"
        ]
        if blocking_gaps and not field_group_work_complete:
            reasons.append(f"{len(blocking_gaps)} blocking coverage gap(s) remain.")
        if state.runtime_status == "running":
            # Dynamic runtime must establish the task contract before it can
            # interpret an apparently empty coverage report as complete.
            if state.task_plan is None:
                reasons.append("Dynamic runtime has not created a task plan yet.")
            if state.dynamic_extraction_plan is None:
                reasons.append("Dynamic runtime has not created a dynamic extraction schema yet.")
            if state.runtime_requires_source_discovery:
                successful_tools = {
                    str(item.get("tool_name"))
                    for item in getattr(state, "tool_result_history", [])
                    if isinstance(item, dict) and item.get("status") in {"completed", "partial"}
                }
                if "discover_sources" not in successful_tools:
                    reasons.append("Dynamic runtime has not completed source discovery yet.")
                if "plan_multi_source_search" not in successful_tools:
                    reasons.append("Dynamic runtime has not planned the multi-source search yet.")
                if not successful_tools.intersection({"search_sources", "search_more"}):
                    reasons.append("Dynamic runtime has not executed a multi-source search attempt yet.")
        return StopDecision(allowed=not reasons, reasons=reasons)


def _field_group_work_complete(state: AgentState) -> bool:
    """Apply the same batch-readiness guard used by the runtime scheduler."""

    coverage = state.coverage_report
    if (
        not coverage.field_groups
        or coverage.unprocessed_relevant_artifacts
        or not all(
            group.initial_search_completed
            and group.status in {"sufficient", "exhausted"}
            for group in coverage.field_groups
        )
    ):
        return False
    completed = {
        str(item.get("tool_name"))
        for item in state.tool_result_history
        if isinstance(item, dict)
        and item.get("status") in {"completed", "partial", "skipped"}
        and int(item.get("workflow_revision") or 0) == state.workflow_revision
    }
    if state.runtime_requires_source_discovery:
        if not completed.intersection({"search_sources", "search_more"}):
            return False
        if state.source_selection_plan is None or "triage_sources" not in completed:
            return False
        if state.runtime_auto_download_sources:
            providers = {
                str(decision.provider or "").strip().casefold()
                for decision in state.source_triage_decisions
                if decision.should_ingest
            }
            if any(provider != "arxiv" for provider in providers) and "ingest_sources" not in completed:
                return False
            if "arxiv" in providers and "ingest_arxiv_pdfs" not in completed:
                return False
    # Local import avoids coupling module initialization while keeping stop
    # validation aligned with the extraction scheduler's fingerprint logic.
    from scidata_agent.agent.action_executor import next_required_derived_stage

    return next_required_derived_stage(state) is None
