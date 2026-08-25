from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from scidata_agent.agent.schemas import (
    ArtifactActionPlan,
    CoverageReport,
    DynamicExtractionPlan,
    QualityReport,
    SourceArtifact,
    SourceCatalogEntry,
)
from scidata_agent.llm.client import LLMCallError
from scidata_agent.llm.nodes import QwenAgentNodes
import scidata_agent.llm.nodes as nodes_module


def make_catalog() -> list[SourceCatalogEntry]:
    return [
        SourceCatalogEntry(
            source_id="source_paper_1",
            title="A paper about the requested experiment",
            source_type="paper",
            provider="arxiv",
            relevance_score=0.91,
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact_pdf_1",
                    source_id="source_paper_1",
                    artifact_type="pdf",
                    local_path="C:/data/paper.pdf",
                    status="downloaded",
                ),
                SourceArtifact(
                    artifact_id="artifact_fig_1",
                    source_id="source_paper_1",
                    artifact_type="image",
                    local_path="C:/data/figure-1.png",
                    status="discovered",
                ),
            ],
        )
    ]


class RecordingClient:
    def __init__(self, payload: Any):
        self.payload = payload
        self.calls: list[tuple[str, str, str, float]] = []

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        self.calls.append((node, system_prompt, user_prompt, temperature))
        return self.payload


def valid_plan_payload() -> dict[str, Any]:
    return {
        "research_goal": "Compare the experimental evidence.",
        "iteration": 2,
        "should_continue": True,
        "actions": [
            {
                "action_id": "action_001",
                "artifact_id": "artifact_pdf_1",
                "action": "parse_pdf_sections",
                "purpose": "Extract the setup and reported results.",
                "expected_fields": ["experimental_setup", "results"],
                "priority": "high",
                "reason": "The downloaded full paper is the primary evidence source.",
                "parameters": {"max_pages": 12},
            },
            {
                "action_id": "action_002",
                "artifact_id": "artifact_fig_1",
                "action": "parse_figure",
                "purpose": "Check whether the figure contains usable quantitative data.",
                "expected_fields": ["figure_data"],
                "priority": "medium",
                "reason": "The figure may contain evidence not present in the text.",
                "parameters": {},
            },
        ],
        "notes": [],
    }


def test_planner_keeps_real_artifact_ids_and_passes_context_to_llm() -> None:
    client = RecordingClient(valid_plan_payload())
    nodes = QwenAgentNodes(client)
    dynamic_plan = DynamicExtractionPlan(
        research_goal="Compare the experimental evidence.",
        dynamic_tables=[
            {
                "table_name": "experiment_results",
                "fields": [{"name": "results", "evidence_required": True}],
            }
        ],
    )

    plan = nodes.plan_artifact_actions(
        "Compare the experimental evidence.",
        make_catalog(),
        dynamic_plan=dynamic_plan,
        quality_report=QualityReport(
            record_count=4,
            issue_count=2,
            warning_count=2,
            evidence_coverage=0.5,
            value_evidence_coverage=0.25,
            notes=["Two requested fields still lack evidence."],
        ),
        processing_log=["source_discovery completed"],
        connector_failures=[{"connector": "arxiv", "error": "temporary timeout"}],
        iteration=2,
    )

    assert plan.actions[0].artifact_id == "artifact_pdf_1"
    assert plan.actions[1].artifact_id == "artifact_fig_1"
    assert len(client.calls) == 1
    node, _system, user_prompt, temperature = client.calls[0]
    assert node == "qwen_artifact_action_planner"
    assert "artifact_pdf_1" in user_prompt
    assert "artifact_fig_1" in user_prompt
    assert "experiment_results" in user_prompt
    assert "Two requested fields still lack evidence" in user_prompt
    assert "temporary timeout" in user_prompt
    assert temperature == 0.05


def test_stop_plan_has_no_artifact_and_stops() -> None:
    payload = {
        "research_goal": "The evidence is sufficient.",
        "should_continue": False,
        "stop_reason": "All required fields have source evidence.",
        "actions": [
            {
                "action_id": "action_stop",
                "artifact_id": None,
                "action": "stop",
                "purpose": "Finish the workflow.",
                "reason": "The available evidence is sufficient.",
            }
        ],
    }
    plan = QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
        "The evidence is sufficient.", make_catalog()
    )

    assert plan.should_continue is False
    assert plan.actions[0].artifact_id is None


def test_stop_plan_is_rejected_when_coverage_is_incomplete() -> None:
    payload = {
        "research_goal": "The evidence is sufficient.",
        "should_continue": False,
        "stop_reason": "The model decided to stop early.",
        "actions": [
            {
                "action_id": "action_stop",
                "artifact_id": None,
                "action": "stop",
                "purpose": "Finish the workflow.",
                "reason": "The available evidence is sufficient.",
            }
        ],
    }
    plan = QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
        "The evidence is sufficient.",
        make_catalog(),
        coverage_report=CoverageReport(
            decision="continue",
            missing_requirements=["experimental setup"],
            reasons=["Required fields are missing or partial: experimental setup."],
        ),
    )

    assert plan.should_continue is True
    assert plan.actions == []
    assert "Stop rejected by coverage auditor" in (plan.stop_reason or "")


def test_unknown_artifact_id_is_dropped_without_losing_valid_actions() -> None:
    payload = valid_plan_payload()
    payload["actions"][0]["artifact_id"] = "artifact_does_not_exist"

    plan = QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
        "Compare the experimental evidence.", make_catalog()
    )

    assert [action.artifact_id for action in plan.actions] == ["artifact_fig_1"]
    assert any("unknown artifact_id" in note for note in plan.notes)


def test_invalid_action_name_is_rejected() -> None:
    payload = valid_plan_payload()
    payload["actions"][0]["action"] = "invent_a_parser"

    with pytest.raises(ValidationError):
        QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
            "Compare the experimental evidence.", make_catalog()
        )


def test_artifact_action_without_artifact_id_is_dropped() -> None:
    payload = valid_plan_payload()
    payload["actions"][0]["artifact_id"] = None

    plan = QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
        "Compare the experimental evidence.", make_catalog()
    )

    assert [action.artifact_id for action in plan.actions] == ["artifact_fig_1"]
    assert any("missing or unknown artifact_id" in note for note in plan.notes)


def test_only_unknown_artifact_actions_stop_bounded_loop_but_allow_pipeline() -> None:
    payload = valid_plan_payload()
    payload["actions"] = [payload["actions"][0]]
    payload["actions"][0]["artifact_id"] = "file_hallucinated"

    plan = QwenAgentNodes(RecordingClient(payload)).plan_artifact_actions(
        "Compare the experimental evidence.", make_catalog()
    )

    assert plan.actions == []
    assert plan.should_continue is False
    assert "normal content pipeline" in (plan.stop_reason or "")


class AlwaysFailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        self.calls += 1
        raise RuntimeError("simulated timeout")


def test_planner_failure_retries_then_propagates_without_fake_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes_module.time, "sleep", lambda _seconds: None)
    client = AlwaysFailingClient()

    with pytest.raises(LLMCallError, match="qwen_artifact_action_planner"):
        QwenAgentNodes(client).plan_artifact_actions(
            "Compare the experimental evidence.", make_catalog()
        )

    assert client.calls == 2
