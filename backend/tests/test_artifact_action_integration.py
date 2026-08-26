from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.llm.client import QwenBailianClient


class IntegrationQwenClient(QwenBailianClient):
    def __init__(self) -> None:
        super().__init__(api_key="integration-mock", model="qwen-integration-mock")

    @property
    def configured(self) -> bool:
        return True

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_task_planner":
            return {
                "domain": "general science",
                "research_goal": "test integration",
                "target_fields": ["metric_name", "metric_value"],
                "dynamic_schema": {},
                "source_requirements": [],
                "validation_rules": [],
                "output_format": ["csv", "json"],
                "need_provenance": True,
                "assumptions": [],
                "schema_notes": [],
            }
        if node == "qwen_dynamic_schema_planner":
            return {
                "research_goal": "test integration",
                "domain": "general science",
                "task_type": "literature_survey",
                "user_focus": [],
                "source_requirements": [],
                "information_needs": [],
                "dynamic_tables": [
                    {
                        "table_name": "test_results",
                        "fields": [{"name": "value", "required": True, "evidence_required": True}],
                    }
                ],
                "quality_rules": [],
                "missing_data_policy": "Use null for missing information.",
            }
        if node == "qwen_source_discovery":
            return {
                "research_goal": "test integration",
                "domain": "general science",
                "recommended_keywords": [],
                "target_data_types": [],
                "dynamic_schema": {},
                "candidate_sources": [],
                "notes": [],
            }
        if node == "qwen_artifact_action_planner":
            return {
                "research_goal": "test integration",
                "iteration": 0,
                "should_continue": False,
                "stop_reason": "Bounded integration smoke test.",
                "actions": [
                    {
                        "action_id": "action_stop",
                        "artifact_id": None,
                        "action": "stop",
                        "purpose": "End the bounded test iteration.",
                        "expected_fields": [],
                        "priority": "low",
                        "reason": "No artifacts are present in this smoke test.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return []


class CSVActionQwenClient(IntegrationQwenClient):
    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            artifact_match = re.search(r'"artifact_id":\s*"(artifact_[a-f0-9]+)"', user_prompt)
            assert artifact_match, "the planner prompt must contain the uploaded CSV artifact ID"
            return {
                "research_goal": "test CSV action",
                "iteration": 0,
                "should_continue": True,
                "stop_reason": None,
                "actions": [
                    {
                        "action_id": "action_parse_csv",
                        "artifact_id": artifact_match.group(1),
                        "action": "parse_csv",
                        "purpose": "Read the uploaded structured results.",
                        "expected_fields": ["metric_value"],
                        "priority": "high",
                        "reason": "The artifact is a local CSV and directly contains requested evidence.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class IteratingQwenClient(IntegrationQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.artifact_planner_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            iteration = self.artifact_planner_calls
            self.artifact_planner_calls += 1
            action = "validate_evidence" if iteration == 0 else "stop"
            return {
                "research_goal": "test iterative artifact actions",
                "iteration": iteration,
                "should_continue": iteration == 0,
                "stop_reason": None if iteration == 0 else "The second iteration completed the bounded test.",
                "actions": [
                    {
                        "action_id": f"action_stop_{iteration}",
                        "artifact_id": None,
                        "action": action,
                        "purpose": "Record the end of this test iteration.",
                        "expected_fields": [],
                        "priority": "low",
                        "reason": "The integration test uses stop as a harmless global action.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


def test_main_agent_runs_bounded_artifact_planner_and_exports_results(tmp_path: Path) -> None:
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=IntegrationQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    result = agent.run(
        "Extract perovskite efficiency and preparation evidence.",
        files=[],
        auto_fetch_arxiv=False,
    )

    assert result.status == "partial"
    assert result.coverage_report.decision == "continue"
    assert result.artifact_action_plan is not None
    assert result.artifact_action_plan.should_continue is True
    assert result.artifact_action_plan.actions == []
    assert result.artifact_action_results == []
    assert "Stop rejected by coverage auditor" in (result.artifact_action_plan.stop_reason or "")
    assert any("artifact_action_planning" in line for line in result.processing_log)
    assert any("artifact_action_execution" in line for line in result.processing_log)

    result_json = Path(result.export_files.json_file)
    action_plan_json = Path(result.export_files.artifact_action_plan_json)
    action_results_json = Path(result.export_files.artifact_action_results_json)
    action_history_json = Path(result.export_files.artifact_action_history_json)
    assert result_json.exists()
    assert action_plan_json.exists()
    assert action_results_json.exists()
    assert action_history_json.exists()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["artifact_action_plan"]["actions"] == []
    assert payload["artifact_action_results"] == []
    assert payload["artifact_action_history"]
    assert all(not item["results"] for item in payload["artifact_action_history"])
    assert len(json.loads(action_history_json.read_text(encoding="utf-8"))) >= 1


def test_main_agent_does_not_duplicate_csv_after_artifact_action(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("method,RMSE\nmodel-a,1.2\nmodel-b,0.8\n", encoding="utf-8")
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=CSVActionQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    result = agent.run(
        "Extract model RMSE values from the uploaded CSV.",
        files=[csv_path],
        auto_fetch_arxiv=False,
    )

    assert result.status == "partial"
    assert result.coverage_report.decision == "continue"
    assert result.coverage_report.missing_requirements == ["value"]
    assert result.artifact_action_results[0].action == "parse_csv"
    assert result.artifact_action_results[0].status == "skipped"
    assert result.artifact_action_history[0].results[0].status == "completed"
    assert result.artifact_action_history[1].results[0].status == "skipped"
    assert result.summary.tables_processed == 1
    assert len(result.source_catalog) == 1


def test_main_agent_preserves_bounded_artifact_action_iterations(tmp_path: Path) -> None:
    client = IteratingQwenClient()
    csv_path = tmp_path / "iteration_input.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=client,
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    result = agent.run(
        "Run two bounded artifact planning iterations.",
        files=[csv_path],
        auto_fetch_arxiv=False,
        max_artifact_action_iterations=2,
    )

    assert result.status == "partial"
    assert result.coverage_report.decision == "continue"
    assert client.artifact_planner_calls == 2
    assert len(result.artifact_action_history) == 2
    assert [item.iteration for item in result.artifact_action_history] == [0, 1]
    assert result.artifact_action_history[0].plan.should_continue is True
    assert result.artifact_action_history[1].plan.should_continue is True
    assert "Stop rejected by coverage auditor" in (
        result.artifact_action_history[1].plan.stop_reason or ""
    )
    planning_steps = [
        index for index, line in enumerate(result.processing_log)
        if line.startswith("Qwen Artifact Action Planner completed")
    ]
    parsing_steps = [
        index for index, line in enumerate(result.processing_log)
        if line.startswith("Source parsing completed")
    ]
    quality_steps = [
        index for index, line in enumerate(result.processing_log)
        if line.startswith("Quality validation completed")
    ]
    assert len(planning_steps) == 2
    assert parsing_steps
    assert quality_steps
    assert planning_steps[1] > quality_steps[0] > parsing_steps[0]
    history_payload = json.loads(
        Path(result.export_files.artifact_action_history_json).read_text(encoding="utf-8")
    )
    assert [item["iteration"] for item in history_payload] == [0, 1]
    result_payload = json.loads(Path(result.export_files.json_file).read_text(encoding="utf-8"))
    assert [item["iteration"] for item in result_payload["artifact_action_history"]] == [0, 1]
