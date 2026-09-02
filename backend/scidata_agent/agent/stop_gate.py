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
        if coverage.decision != "allow_stop":
            reasons.append(
                f"Coverage gate is {coverage.decision!r}; required evidence is not complete."
            )
        if coverage.missing_requirements:
            reasons.append(
                "Missing requirements: " + ", ".join(coverage.missing_requirements[:12])
            )
        blocking_gaps = [gap for gap in coverage.gaps if gap.priority != "low"]
        if blocking_gaps:
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
