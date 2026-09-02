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


class WorkflowActionQwenClient(IntegrationQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.artifact_planner_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            call_number = self.artifact_planner_calls
            self.artifact_planner_calls += 1
            if call_number == 0:
                action = "plan_task"
                purpose = "Create the task contract before selecting content tools."
                expected_fields = []
            elif call_number == 1:
                action = "plan_dynamic_schema"
                purpose = "Create the task-specific extraction schema before parsing evidence."
                expected_fields = []
            else:
                action = "parse_content"
                purpose = "Run content extraction over the uploaded evidence."
                expected_fields = ["metric_value"]
            return {
                "research_goal": "run the content workflow tool",
                "iteration": call_number,
                "should_continue": True,
                "stop_reason": None,
                "actions": [
                    {
                        "action_id": f"workflow_{action}_{call_number}",
                        "artifact_id": None,
                        "action": action,
                        "purpose": purpose,
                        "expected_fields": expected_fields,
                        "priority": "high",
                        "reason": "Follow the dynamic initialization and extraction workflow.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class DownloadOnlyQwenClient(IntegrationQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.artifact_planner_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            call_number = self.artifact_planner_calls
            self.artifact_planner_calls += 1
            if call_number == 0:
                action = "plan_task"
                artifact_id = None
                purpose = "Create the task contract before selecting content tools."
            elif call_number == 1:
                action = "plan_dynamic_schema"
                artifact_id = None
                purpose = "Create the task-specific extraction schema before using the artifact."
            else:
                artifact_match = re.search(r'"artifact_id":\s*"(artifact_[a-f0-9]+)"', user_prompt)
                assert artifact_match, "the planner prompt must contain the uploaded artifact ID"
                action = "download_artifact"
                artifact_id = artifact_match.group(1)
                purpose = "Materialize the selected research artifact."
            return {
                "research_goal": "test explicit download step",
                "iteration": call_number,
                "should_continue": True,
                "stop_reason": None,
                "actions": [
                    {
                        "action_id": f"action_{action}_{call_number}",
                        "artifact_id": artifact_id,
                        "action": action,
                        "purpose": purpose,
                        "expected_fields": [],
                        "priority": "high",
                        "reason": "Exercise a download-only dynamic turn.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class GranularWorkflowQwenClient(IntegrationQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.artifact_planner_calls = 0
        self.actions = [
            "plan_task",
            "plan_dynamic_schema",
            "parse_source_content",
            "extract_figures",
            "interpret_sections",
            "extract_dynamic_records",
            "extract_records",
        ]

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            call_number = self.artifact_planner_calls
            self.artifact_planner_calls += 1
            action = self.actions[min(call_number, len(self.actions) - 1)]
            return {
                "research_goal": "run granular content workflow tools",
                "iteration": call_number,
                "should_continue": True,
                "stop_reason": None,
                "actions": [
                    {
                        "action_id": f"granular_{action}_{call_number}",
                        "artifact_id": None,
                        "action": action,
                        "purpose": f"Run the {action} stage selected for this evidence gap.",
                        "expected_fields": ["metric_value"] if action in {
                            "extract_dynamic_records",
                            "extract_records",
                        } else [],
                        "priority": "high",
                        "reason": "Exercise independently selectable content stages.",
                        "parameters": {},
                    }
                ],
                "notes": [],
            }
        return super().generate_json(node, system_prompt, user_prompt, temperature=temperature)


class ResumableWorkflowQwenClient(IntegrationQwenClient):
    def __init__(self) -> None:
        super().__init__()
        self.artifact_planner_calls = 0

    def generate_json(self, node: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Any:
        if node == "qwen_artifact_action_planner":
            self.artifact_planner_calls += 1
            call_number = self.artifact_planner_calls - 1
            if call_number == 0:
                action = "plan_task"
                should_continue = True
                stop_reason = None
            elif call_number == 1:
                action = "plan_dynamic_schema"
                should_continue = True
                stop_reason = None
            elif call_number == 2:
                action = "parse_content"
                should_continue = True
                stop_reason = None
            else:
                action = "stop"
                should_continue = False
                stop_reason = "Resume smoke test reached its second decision."
            return {
                "research_goal": "resume dynamic workflow",
                "iteration": call_number,
                "should_continue": should_continue,
                "stop_reason": stop_reason,
                "actions": [
                    {
                        "action_id": f"resume_action_{self.artifact_planner_calls}",
                        "artifact_id": None,
                        "action": action,
                        "purpose": "Exercise checkpoint-backed dynamic execution.",
                        "expected_fields": ["metric_value"] if action == "parse_content" else [],
                        "priority": "high" if action != "stop" else "low",
                        "reason": "Initialize, perform work, then exercise checkpoint-backed resume.",
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
    assert result.runtime_iteration >= 1
    assert result.agent_decision_history
    assert any(event["event_type"] == "agent_decision" for event in result.agent_trace)

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
    assert Path(result.export_files.agent_trace_json).exists()
    assert Path(result.export_files.decision_history_json).exists()
    assert Path(result.export_files.tool_history_json).exists()


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


def test_dynamic_runtime_executes_workflow_tool_for_uploaded_content(tmp_path: Path) -> None:
    csv_path = tmp_path / "workflow_input.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=WorkflowActionQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    result = agent.run(
        "Extract the metric value from the uploaded result file.",
        files=[csv_path],
        auto_fetch_arxiv=False,
        enable_live_search=False,
        enable_dynamic_runtime=True,
        max_agent_iterations=3,
    )

    assert [
        item.plan.actions[0].action for item in result.artifact_action_history
    ] == ["plan_task", "plan_dynamic_schema", "parse_content"]
    assert result.artifact_action_results[0].status == "completed"
    assert any(
        event.get("event_type") == "tool_completed"
        and event.get("tool_name") == "parse_content"
        for event in result.agent_trace
    )
    assert result.tool_result_history[-1]["tool_name"] == "parse_content"


def test_dynamic_runtime_does_not_implicitly_run_content_pipeline_after_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "download_only.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=DownloadOnlyQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    def unexpected_implicit_pipeline(*args, **kwargs):
        raise AssertionError("dynamic download must not trigger the legacy content pipeline")

    monkeypatch.setattr(agent, "_run_content_pipeline", unexpected_implicit_pipeline)
    result = agent.run(
        "Materialize the uploaded research artifact before deciding how to parse it.",
        files=[csv_path],
        auto_fetch_arxiv=False,
        enable_live_search=False,
        enable_dynamic_runtime=True,
        max_agent_iterations=3,
    )

    assert result.runtime_status == "partial"
    assert [item["tool_name"] for item in result.tool_result_history] == [
        "plan_task",
        "plan_dynamic_schema",
        "download_artifact",
    ]
    assert result.summary.tables_processed == 0


def test_dynamic_runtime_can_select_granular_content_stages(tmp_path: Path) -> None:
    csv_path = tmp_path / "granular_input.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    client = GranularWorkflowQwenClient()
    result = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=client,
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    ).run(
        "Extract evidence through independently selectable content stages.",
        files=[csv_path],
        auto_fetch_arxiv=False,
        enable_live_search=False,
        enable_dynamic_runtime=True,
        max_agent_iterations=len(client.actions),
    )

    assert [
        item.plan.actions[0].action for item in result.artifact_action_history
    ] == client.actions
    assert [item["tool_name"] for item in result.tool_result_history] == client.actions
    assert all(action != "parse_content" for action in client.actions)
    assert result.runtime_status == "partial"


def test_dynamic_runtime_repeats_decision_turns_until_safety_budget(tmp_path: Path) -> None:
    agent = SciDataAgent(
        output_dir=tmp_path / "outputs",
        llm_client=IntegrationQwenClient(),
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    )

    result = agent.run(
        "Use the dynamic runtime to inspect available scientific evidence.",
        files=[],
        auto_fetch_arxiv=False,
        enable_dynamic_runtime=True,
        max_agent_iterations=2,
    )

    assert result.status == "partial"
    assert result.runtime_status == "partial"
    assert result.runtime_iteration == 2
    assert len(result.agent_decision_history) == 2
    assert len([event for event in result.agent_trace if event["event_type"] == "agent_decision"]) == 2
    assert "safety budget" in (result.runtime_stop_reason or "")
    assert any("task plan" in reason for reason in result.stop_rejections)


def test_dynamic_runtime_resumes_from_tool_checkpoint_without_reexecution(tmp_path: Path) -> None:
    csv_path = tmp_path / "resume_input.csv"
    csv_path.write_text("metric,value\nrmse,0.8\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    client = ResumableWorkflowQwenClient()
    options = {
        "files": [csv_path],
        "auto_fetch_arxiv": False,
        "enable_live_search": False,
        "enable_dynamic_runtime": True,
        "max_agent_iterations": 3,
        "task_id": "resume_dynamic_test",
    }

    first = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    ).run("Resume the dynamic workflow from its last tool result.", **options)

    assert first.status == "partial"
    assert first.runtime_iteration == 3
    assert len(first.tool_result_history) == 3
    assert [item["tool_name"] for item in first.tool_result_history] == [
        "plan_task",
        "plan_dynamic_schema",
        "parse_content",
    ]
    checkpoint = output_dir / "resume_dynamic_test" / "agent_checkpoint.json"
    assert checkpoint.exists()

    second = SciDataAgent(
        output_dir=output_dir,
        llm_client=client,
        require_llm=True,
        monitor_console=False,
        monitor_enabled=False,
    ).run(
        "Resume the dynamic workflow from its last tool result.",
        resume=True,
        **options,
    )

    assert second.status == "partial"
    # The configured budget is task-global. A resume cannot silently grant a
    # fresh batch of turns after the original three were consumed.
    assert second.runtime_iteration == 3
    assert len(second.tool_result_history) == 3
    assert client.artifact_planner_calls == 3
    assert any("Resuming from checkpoint" in line for line in second.processing_log)


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
