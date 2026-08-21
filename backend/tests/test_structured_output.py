from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import ValidationError

from scidata_agent.agent.schemas import (
    AgentState,
    ArtifactActionPlan,
    DynamicExtractionPlan,
    SourceSelectionPlan,
    TaskPlan,
    UploadedFile,
)
from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.llm.nodes import QwenAgentNodes
from scidata_agent.llm.structured_output import normalize_payload_for_model


class _ConfiguredClient:
    configured = True


def test_schema_driven_normalization_repairs_common_shape_drift() -> None:
    payload = {
        "research_goal": "select relevant sources",
        "decisions": [
            {
                "source_id": "src-1",
                "decision": "deep read",
                "priority": "HIGH",
                "priority_score": "0.92",
                "reason": "matches the research question",
                "matched_requirements": "architecture and evaluation",
                "expected_extractable_fields": ["architecture"],
                "risk_notes": None,
            }
        ],
        "notes": "one source needs inspection",
    }

    normalized, events = normalize_payload_for_model(payload, SourceSelectionPlan)
    plan = SourceSelectionPlan.model_validate(normalized)

    assert plan.decisions[0].decision == "deep_read"
    assert plan.decisions[0].priority == "high"
    assert abs(plan.decisions[0].priority_score - 0.92) < 1e-9
    assert plan.decisions[0].matched_requirements == ["architecture and evaluation"]
    assert plan.notes == ["one source needs inspection"]
    assert {event["rule"] for event in events} >= {
        "literal_token_normalization",
        "numeric_string_to_float",
        "scalar_to_list",
    }


def test_schema_driven_normalization_recurses_through_nested_models() -> None:
    payload = {
        "research_goal": "extract experiments",
        "dynamic_tables": "[{\"table_name\": \"results\", \"priority\": \"HIGH\", \"fields\": {\"name\": \"score\", \"required\": \"true\", \"examples\": \"0.91\"}}]",
        "information_needs": {"need_name": "evaluation", "priority": "low"},
    }

    normalized, events = normalize_payload_for_model(payload, DynamicExtractionPlan)
    plan = DynamicExtractionPlan.model_validate(normalized)

    assert plan.dynamic_tables[0].priority == "high"
    assert plan.dynamic_tables[0].fields[0].name == "score"
    assert plan.dynamic_tables[0].fields[0].required is True
    assert plan.dynamic_tables[0].fields[0].examples == ["0.91"]
    assert plan.information_needs[0].need_name == "evaluation"
    assert len(events) >= 6


def test_normalization_events_are_recorded_by_agent_nodes() -> None:
    nodes = QwenAgentNodes(_ConfiguredClient())
    payload = {
        "research_goal": "extract experiments",
        "iteration": "2",
        "should_continue": "false",
        "actions": [],
        "notes": "done",
    }

    normalized = nodes._normalize_payload("test_node", payload, ArtifactActionPlan)

    assert normalized["iteration"] == 2
    assert normalized["should_continue"] is False
    assert nodes.normalization_events
    assert all(event["node"] == "test_node" for event in nodes.normalization_events)
    assert {"path", "rule", "original_type", "normalized_type"}.issubset(
        nodes.normalization_events[0]
    )


def test_ambiguous_values_are_not_silently_discarded() -> None:
    payload = {
        "research_goal": "select sources",
        "decisions": [
            {
                "source_id": "src-1",
                "decision": "not-a-supported-action",
                "reason": "unclear",
            }
        ],
    }

    normalized, _ = normalize_payload_for_model(payload, SourceSelectionPlan)

    try:
        SourceSelectionPlan.model_validate(normalized)
    except ValidationError:
        pass
    else:
        raise AssertionError("unsupported Literal value was silently accepted")


def test_agent_processing_log_records_normalization_events() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        agent = SciDataAgent(
            output_dir=tmp_path,
            llm_client=_ConfiguredClient(),
            monitor_console=False,
            monitor_enabled=False,
        )
        state = AgentState(
            task_id="normalization-log-test",
            research_question="test schema normalization",
            files=[UploadedFile(filename="paper.pdf", path=tmp_path / "paper.pdf")],
            output_dir=tmp_path,
        )
        agent.llm_nodes.normalization_events.append(
            {
                "node": "test_node",
                "path": "$.notes",
                "rule": "scalar_to_list",
                "original_type": "str",
                "normalized_type": "list",
            }
        )

        agent._append_normalization_log(state, "task_planning", 0)

        assert state.processing_log == [
            "LLM output normalization: step=task_planning, events=1, paths=[$.notes]."
        ]
