"""Regression check for grouped retrieval and deterministic completion.

Run from the backend directory with the project's conda environment:
    conda run -n scidata-agent python scripts/check_field_group_workflow.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import scidata_agent.agent.action_executor as action_executor_module
from scidata_agent.agent.action_executor import ArtifactActionExecutor
from scidata_agent.agent.scidata_agent import (
    SciDataAgent,
    _field_group_work_complete,
    _required_action_parameters,
)
from scidata_agent.agent.schemas import (
    AgentState,
    ArtifactAction,
    DynamicExtractionPlan,
    DynamicFieldSpec,
    DynamicRecord,
    DynamicTableSpec,
    MultiSourceSearchPlan,
    SourceDiscoveryPlan,
    SourceSearchRequest,
    SourceSelectionPlan,
    TaskPlan,
)
from scidata_agent.llm.nodes import _ensure_field_group_search_requests
from scidata_agent.tools.coverage import build_coverage_report


def main() -> None:
    groups = [
        {
            "group_id": "performance",
            "label": "performance",
            "fields": ["peak_magnitude"],
        },
        {
            "group_id": "decline",
            "label": "decline",
            "fields": ["decline_rate"],
        },
    ]
    raw_plan = MultiSourceSearchPlan(
        research_goal="Compare light curves.",
        should_search=False,
        search_requests=[
            SourceSearchRequest(
                connector_name="openalex",
                source_type="paper_metadata",
                query="light curve peak magnitude",
                field_group_id="performance",
                target_fields=["peak_magnitude"],
            )
        ],
    )
    search_plan = _ensure_field_group_search_requests(
        raw_plan,
        "Compare light curves.",
        groups,
    )
    request_groups = {
        request.field_group_id for request in search_plan.search_requests
    }
    assert search_plan.should_search is True
    assert request_groups == {"performance", "decline"}

    with TemporaryDirectory(prefix="scidata-field-groups-") as temp_dir:
        state = AgentState(
            research_question="Compare light curves.",
            files=[],
            output_dir=Path(temp_dir),
            task_plan=TaskPlan(
                research_goal="Compare light curves.",
                target_fields=["peak_magnitude", "decline_rate"],
            ),
            dynamic_extraction_plan=DynamicExtractionPlan(
                research_goal="Compare light curves.",
                dynamic_tables=[
                    DynamicTableSpec(
                        table_name="performance",
                        fields=[
                            DynamicFieldSpec(name="peak_magnitude", required=True)
                        ],
                    ),
                    DynamicTableSpec(
                        table_name="decline",
                        fields=[
                            DynamicFieldSpec(name="decline_rate", required=True)
                        ],
                    ),
                ],
            ),
            source_discovery_plan=SourceDiscoveryPlan(
                research_goal="Compare light curves."
            ),
            multi_source_search_plan=search_plan,
            source_selection_plan=SourceSelectionPlan(
                research_goal="Compare light curves."
            ),
            runtime_requires_source_discovery=True,
            runtime_group_initial_searches=["performance", "decline"],
            runtime_group_search_more_counts={"decline": 1},
            runtime_search_more_limit=2,
            tool_result_history=[
                {
                    "tool_name": "search_sources",
                    "status": "completed",
                    "workflow_revision": 0,
                },
                {
                    "tool_name": "triage_sources",
                    "status": "completed",
                    "workflow_revision": 0,
                },
            ],
            dynamic_records=[
                DynamicRecord(
                    table_name="performance",
                    fields={"peak_magnitude": value},
                    source_file=f"paper-{index}.pdf",
                )
                for index, value in enumerate((-18.1, -18.4, -17.9), start=1)
            ],
        )

        state.coverage_report = build_coverage_report(state)
        assert [
            group.status for group in state.coverage_report.field_groups
        ] == ["sufficient", "insufficient"]
        assert SciDataAgent._required_dynamic_workflow_actions(state) == [
            "search_more"
        ]
        parameters = _required_action_parameters(state, "search_more")
        assert parameters["field_group_id"] == "decline"
        assert parameters["target_fields"] == ["decline_rate"]

        pending_state = state.model_copy(deep=True)
        pending_state.runtime_group_initial_searches = ["performance"]
        pending_state.runtime_group_search_more_counts = {"decline": 2}
        pending_state.coverage_report = build_coverage_report(pending_state)
        pending_parameters = _required_action_parameters(
            pending_state,
            "search_more",
        )
        assert pending_parameters["field_group_id"] == "decline"
        assert pending_parameters["initial_group_search"] is True

        state.runtime_group_search_more_counts["decline"] = 2
        state.coverage_report = build_coverage_report(state)
        assert [
            group.status for group in state.coverage_report.field_groups
        ] == ["sufficient", "exhausted"]
        assert _field_group_work_complete(state) is True
        assert SciDataAgent._required_dynamic_workflow_actions(state) == ["stop"]
        assert state.coverage_report.decision == "allow_stop"

        # The aggregate audit count may exceed two, while each group still
        # owns an independent two-attempt limit.
        class Planner:
            def plan_multi_source_search(self, *_args, **_kwargs):
                return MultiSourceSearchPlan(
                    research_goal="Compare light curves.",
                    search_requests=[
                        SourceSearchRequest(
                            connector_name="openalex",
                            source_type="paper_metadata",
                            query="decline rate evidence",
                        )
                    ],
                )

        original_search = action_executor_module.execute_multi_source_search
        action_executor_module.execute_multi_source_search = lambda _plan, **_kwargs: (
            [],
            {
                "status": "completed",
                "searched": 1,
                "failed": 0,
                "connector_status": [],
            },
        )
        try:
            counter_state = AgentState(
                research_question="Compare light curves.",
                files=[],
                output_dir=Path(temp_dir),
                source_discovery_plan=SourceDiscoveryPlan(
                    research_goal="Compare light curves."
                ),
                runtime_search_more_count=7,
                runtime_search_more_limit=2,
                runtime_group_search_more_counts={"performance": 2},
            )
            result = ArtifactActionExecutor(Planner()).execute_action(
                ArtifactAction(
                    action_id="search-decline",
                    action="search_more",
                    purpose="Fill the decline field group.",
                    reason="The group remains insufficient.",
                    parameters={
                        "field_group_id": "decline",
                        "target_fields": ["decline_rate"],
                    },
                ),
                counter_state,
            )
            assert result.status == "completed"
            assert counter_state.runtime_group_search_more_counts["decline"] == 1
            assert counter_state.runtime_search_more_count == 8
        finally:
            action_executor_module.execute_multi_source_search = original_search

    print("PASS: grouped initial retrieval, bounded search_more, and completion")


if __name__ == "__main__":
    main()
