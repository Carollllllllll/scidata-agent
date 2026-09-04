from __future__ import annotations

import threading
from typing import Any

from scidata_agent.agent.decision import AgentDecision
from scidata_agent.agent.harness import AgentHarness
from scidata_agent.agent.observation import ObservationBuilder
from scidata_agent.agent.policy import AgentPolicy
from scidata_agent.agent.schemas import (
    AgentState,
    CoverageReport,
    CoverageGap,
    DiscoveredSource,
    DynamicExtractionPlan,
    MultiSourceSearchPlan,
    SourceArtifact,
    SourceCatalogEntry,
    SourceDiscoveryPlan,
)
from scidata_agent.agent.stop_gate import StopGate
from scidata_agent.agent.tool_protocol import ToolCall, ToolResult, ToolSpec
from scidata_agent.agent.tool_registry import ToolRegistry
from scidata_agent.agent.tool_runtime import ToolRuntime
from scidata_agent.agent.tool_registry import build_artifact_tool_registry


def make_state(tmp_path) -> AgentState:
    return AgentState(
        research_question="Find evidence for a research claim.",
        files=[],
        output_dir=tmp_path / "outputs",
    )


def test_dynamic_stop_gate_requires_initialization_contract(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    state.coverage_report = CoverageReport(decision="allow_stop")

    result = StopGate().evaluate(
        AgentDecision(decision="stop", reason="Nothing else is visible yet."),
        state,
    )

    assert result.allowed is False
    assert any("task plan" in reason for reason in result.reasons)
    assert any("dynamic extraction schema" in reason for reason in result.reasons)


def test_stop_gate_does_not_block_on_unavailable_gap(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    state.task_plan = {"research_goal": state.research_question}
    state.dynamic_extraction_plan = DynamicExtractionPlan(
        research_goal=state.research_question,
    )
    state.coverage_report = CoverageReport(
        decision="allow_stop",
        gaps=[
            CoverageGap(
                gap_id="unavailable-paper",
                requirement_name="Required paper",
                priority="high",
                status="unavailable",
                reason="No usable source remains.",
            )
        ],
    )

    result = StopGate().evaluate(
        AgentDecision(decision="stop", reason="All actionable work is complete."),
        state,
    )

    assert result.allowed is True


def test_policy_rejects_content_action_before_its_prerequisites(tmp_path) -> None:
    state = make_state(tmp_path)
    decision = AgentDecision(
        decision="continue",
        tool_calls=[
            ToolCall(
                call_id="extract-too-early",
                tool_name="extract_dynamic_records",
            )
        ],
    )

    result = AgentPolicy(build_artifact_tool_registry()).validate(decision, state)

    assert result.allowed is False
    assert any("parsed text or table evidence" in violation for violation in result.violations)
    assert any("dynamic extraction schema" in violation for violation in result.violations)


def test_policy_rejects_continue_without_a_tool_call(tmp_path) -> None:
    state = make_state(tmp_path)

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(decision="continue", reason="I need another turn."),
        state,
    )

    assert result.allowed is False
    assert result.tool_calls == []
    assert result.violations == ["A continue decision must contain at least one tool call."]


def test_harness_retries_policy_rejection_without_consuming_extra_iteration(tmp_path) -> None:
    state = make_state(tmp_path)
    calls: list[str] = []
    registry = ToolRegistry([ToolSpec(name="collect_evidence", description="Collect evidence.")])

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        if not context.stop_rejections:
            return AgentDecision(decision="continue", reason="Repair this decision.")
        return AgentDecision(
            decision="continue",
            reason="Use the valid tool call after the guard feedback.",
            tool_calls=[ToolCall(call_id="collect-1", tool_name="collect_evidence")],
        )

    result = AgentHarness(registry, decide, runtime=runtime).run(state, max_iterations=1)

    assert result.iterations == 1
    assert len(result.decisions) == 2
    assert calls == ["collect-1"]
    assert any(event["event_type"] == "policy_retry" for event in result.trace)


def test_policy_rejects_table_parsing_before_task_contract(tmp_path) -> None:
    state = make_state(tmp_path)
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Local paper",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-1",
                    source_id="source-1",
                    artifact_type="pdf",
                    local_path=str(tmp_path / "paper.pdf"),
                )
            ],
        )
    ]

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(
                    call_id="table-too-early",
                    tool_name="parse_table",
                    arguments={"artifact_id": "artifact-1"},
                )
            ],
        ),
        state,
    )

    assert result.allowed is False
    assert any("completed task plan" in violation for violation in result.violations)
    assert any("dynamic extraction schema" in violation for violation in result.violations)


def test_policy_allows_ordered_initialization_batch(tmp_path) -> None:
    state = make_state(tmp_path)
    decision = AgentDecision(
        decision="continue",
        tool_calls=[
            ToolCall(call_id="plan", tool_name="plan_task"),
            ToolCall(call_id="schema", tool_name="plan_dynamic_schema"),
            ToolCall(call_id="discover", tool_name="discover_sources"),
        ],
    )

    result = AgentPolicy(build_artifact_tool_registry()).validate(decision, state)

    assert result.allowed is True
    assert [call.tool_name for call in result.tool_calls] == [
        "plan_task",
        "plan_dynamic_schema",
        "discover_sources",
    ]


def test_policy_requires_multi_source_search_before_remote_source_processing(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_requires_source_discovery = True
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                source_id="source-1",
                title="Remote paper",
                source_type="paper",
                url="https://example.com/paper.pdf",
            )
        ],
    )
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Remote paper",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-1",
                    source_id="source-1",
                    artifact_type="pdf",
                    url="https://example.com/paper.pdf",
                )
            ],
        )
    ]

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(
                    call_id="download-1",
                    tool_name="download_artifact",
                    arguments={"artifact_id": "artifact-1"},
                )
            ],
        ),
        state,
    )

    assert result.allowed is False
    assert any("multi-source search attempt" in item for item in result.violations)


def test_policy_allows_source_processing_after_partial_search(tmp_path) -> None:
    state = make_state(tmp_path)
    state.task_plan = {"research_goal": state.research_question}
    state.dynamic_extraction_plan = DynamicExtractionPlan(
        research_goal=state.research_question,
    )
    state.runtime_requires_source_discovery = True
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[DiscoveredSource(source_id="source-1", title="Paper", source_type="paper")],
    )
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Paper",
            artifacts=[SourceArtifact(artifact_id="artifact-1", source_id="source-1", artifact_type="pdf", url="https://example.com/paper.pdf")],
        )
    ]
    state.multi_source_search_plan = MultiSourceSearchPlan(research_goal=state.research_question)
    state.tool_result_history = [
        {
            "call_id": "search-1",
            "tool_name": "search_sources",
            "status": "partial",
        }
    ]

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(
                    call_id="download-1",
                    tool_name="download_artifact",
                    arguments={"artifact_id": "artifact-1"},
                )
            ],
        ),
        state,
    )

    assert result.allowed is True
    assert [call.tool_name for call in result.tool_calls] == ["download_artifact"]


def test_dynamic_stop_gate_requires_search_attempt_even_when_catalog_is_empty(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    state.runtime_requires_source_discovery = True
    state.task_plan = {"research_goal": state.research_question}
    state.dynamic_extraction_plan = {"research_goal": state.research_question}
    state.source_discovery_plan = SourceDiscoveryPlan(research_goal=state.research_question)
    state.multi_source_search_plan = MultiSourceSearchPlan(research_goal=state.research_question)
    state.coverage_report = CoverageReport(decision="allow_stop")

    result = StopGate().evaluate(
        AgentDecision(decision="stop", reason="No candidates were returned."),
        state,
    )

    assert result.allowed is False
    assert any("multi-source search attempt" in item for item in result.reasons)


def test_policy_filters_duplicate_call_without_dropping_valid_calls(tmp_path) -> None:
    state = make_state(tmp_path)
    state.task_plan = {
        "research_goal": state.research_question,
        "domain": "science",
        "target_fields": [],
        "source_requirements": [],
        "need_provenance": True,
        "output_format": ["json"],
    }
    state.dynamic_extraction_plan = DynamicExtractionPlan(
        research_goal=state.research_question,
    )
    entry = SourceCatalogEntry(
        source_id="source-1",
        title="paper",
        artifacts=[SourceArtifact(artifact_id="artifact-1", source_id="source-1", artifact_type="pdf")],
    )
    state.source_catalog = [entry]
    first = ToolCall(
        call_id="metadata-1",
        tool_name="read_metadata",
        arguments={"artifact_id": "artifact-1"},
    )
    second = ToolCall(
        call_id="metadata-2",
        tool_name="read_metadata",
        arguments={"artifact_id": "artifact-1"},
    )

    result = AgentPolicy(build_artifact_tool_registry()).validate(
        AgentDecision(decision="continue", tool_calls=[first, second]), state
    )

    assert result.allowed is True
    assert [call.call_id for call in result.tool_calls] == ["metadata-1"]
    assert any("Duplicate tool call" in violation for violation in result.violations)


def test_harness_runs_observe_decide_act_and_records_tool_history(tmp_path) -> None:
    state = make_state(tmp_path)
    state.coverage_report = CoverageReport(decision="allow_stop")
    calls: list[str] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        calls.append(call.call_id)
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"evidence_id": "ev-1"},
            evidence_refs=["ev-1"],
        )

    registry = ToolRegistry([ToolSpec(name="collect_evidence", description="Collect evidence.")])
    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        if not context.tool_result_history:
            return AgentDecision(
                decision="continue",
                reason="Evidence has not been collected yet.",
                tool_calls=[
                    ToolCall(
                        call_id="collect-1",
                        tool_name="collect_evidence",
                        reason="Collect the first evidence item.",
                    )
                ],
            )
        return AgentDecision(decision="stop", reason="The coverage gate is satisfied.")

    result = AgentHarness(registry, decide, runtime=runtime).run(state, max_iterations=3)

    assert result.status == "completed"
    assert result.iterations == 2
    assert calls == ["collect-1"]
    assert len(result.tool_results) == 1
    assert state.tool_result_history[0]["evidence_refs"] == ["ev-1"]
    assert [event["event_type"] for event in result.trace].count("agent_decision") == 2
    assert any(event["event_type"] == "tool_completed" for event in result.trace)


def test_harness_rejects_early_stop_then_continues(tmp_path) -> None:
    state = make_state(tmp_path)
    calls: list[str] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        calls.append(call.call_id)
        state.coverage_report = CoverageReport(decision="allow_stop")
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([ToolSpec(name="search", description="Search sources.")])
    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        if not context.stop_rejections:
            return AgentDecision(decision="stop", reason="Premature stop request.")
        if not context.tool_result_history:
            return AgentDecision(
                decision="continue",
                tool_calls=[ToolCall(call_id="search-1", tool_name="search")],
            )
        return AgentDecision(decision="stop", reason="Coverage is now complete.")

    result = AgentHarness(registry, decide, runtime=runtime).run(state, max_iterations=4)

    assert result.status == "completed"
    assert calls == ["search-1"]
    assert result.stop_rejections
    assert any("Coverage gate" in reason for reason in result.stop_rejections)
    assert result.decisions[0].decision == "stop"
    assert result.decisions[-1].decision == "stop"


def test_harness_stops_after_repeated_identical_stop_rejection(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    state.coverage_report = CoverageReport(decision="allow_stop")

    def decide(observation, context) -> AgentDecision:
        return AgentDecision(decision="stop", reason="No work is visible.")

    result = AgentHarness(
        build_artifact_tool_registry(),
        decide,
    ).run(state, max_iterations=5)

    assert result.status == "partial"
    assert result.terminal is True
    assert result.iterations == 2
    assert "repeatedly requested stop" in result.stop_reason
    assert state.runtime_status == "partial"
    assert any(event["event_type"] == "stop_rejected_repeat" for event in result.trace)


def test_harness_resets_stop_rejection_streak_after_progress(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    registry = ToolRegistry([ToolSpec(name="search", description="Search sources.")])
    runtime = ToolRuntime(
        registry,
        handler=lambda call, context, options: ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
        ),
    )
    decisions = iter([
        AgentDecision(decision="stop", reason="Blocked."),
        AgentDecision(
            decision="continue",
            tool_calls=[ToolCall(call_id="search-1", tool_name="search")],
        ),
        AgentDecision(decision="stop", reason="Blocked again."),
    ])

    result = AgentHarness(registry, lambda observation, context: next(decisions), runtime=runtime).run(
        state,
        max_iterations=3,
    )

    assert result.status == "partial"
    assert result.terminal is False
    assert result.iterations == 3
    assert not any(event["event_type"] == "stop_rejected_repeat" for event in result.trace)


def test_harness_restores_one_stop_rejection_from_checkpoint(tmp_path) -> None:
    state = make_state(tmp_path)
    state.runtime_status = "running"
    state.agent_trace = [
        {
            "event_type": "stop_rejected",
            "payload": {
                "reasons": [
                    "Dynamic runtime has not created a task plan yet.",
                    "Dynamic runtime has not created a dynamic extraction schema yet.",
                ]
            },
        }
    ]
    state.coverage_report = CoverageReport(decision="allow_stop")

    result = AgentHarness(
        build_artifact_tool_registry(),
        lambda observation, context: AgentDecision(decision="stop", reason="Still blocked."),
    ).run(state, max_iterations=1)

    assert result.status == "partial"
    assert result.terminal is True
    assert result.iterations == 1
    assert "task plan" in result.stop_reason


def test_harness_trace_callback_receives_decision_and_tool_lifecycle(tmp_path) -> None:
    state = make_state(tmp_path)
    events: list[dict[str, Any]] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        state.coverage_report = CoverageReport(decision="allow_stop")
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="completed",
            data={"evidence_id": "ev-1"},
            evidence_refs=["ev-1"],
        )

    registry = ToolRegistry([ToolSpec(name="collect_evidence", description="Collect evidence.")])
    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        if not context.tool_result_history:
            return AgentDecision(
                decision="continue",
                tool_calls=[ToolCall(call_id="collect-1", tool_name="collect_evidence")],
            )
        return AgentDecision(decision="stop", reason="Evidence is complete.")

    result = AgentHarness(
        registry,
        decide,
        runtime=runtime,
        trace_callback=lambda event, context: events.append(event),
    ).run(state, max_iterations=3)

    event_types = [event["event_type"] for event in events]
    assert result.status == "completed"
    assert event_types[:4] == [
        "agent_decision",
        "tool_started",
        "tool_completed",
        "tool_result",
    ]
    assert event_types[-1] == "agent_decision"
    assert any(event.get("tool_name") == "collect_evidence" for event in events)


def test_harness_ignores_trace_callback_failures(tmp_path) -> None:
    state = make_state(tmp_path)
    state.coverage_report = CoverageReport(decision="allow_stop")
    registry = ToolRegistry([ToolSpec(name="search", description="Search sources.")])

    def fail_callback(event, context) -> None:
        raise RuntimeError("monitor unavailable")

    def decide(observation, context) -> AgentDecision:
        return AgentDecision(decision="stop", reason="Coverage is complete.")

    result = AgentHarness(registry, decide, trace_callback=fail_callback).run(
        state,
        max_iterations=1,
    )

    assert result.status == "completed"
    assert result.stop_reason == "Coverage is complete."


def test_harness_checkpoints_each_tool_result_before_iteration_finishes(tmp_path) -> None:
    state = make_state(tmp_path)
    checkpoint_snapshots: list[tuple[int, int]] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([ToolSpec(name="search", description="Search sources.")])
    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        return AgentDecision(
            decision="continue",
            tool_calls=[ToolCall(call_id="search-1", tool_name="search")],
        )

    result = AgentHarness(
        registry,
        decide,
        runtime=runtime,
        checkpoint_callback=lambda context: checkpoint_snapshots.append(
            (len(context.tool_result_history), len(context.agent_trace))
        ),
    ).run(state, max_iterations=1)

    assert result.status == "partial"
    assert checkpoint_snapshots == [(1, 4)]
    assert len(state.tool_result_history) == 1
    assert state.agent_trace[-1]["event_type"] == "tool_result"


def test_tool_runtime_parallelizes_an_all_safe_batch_and_preserves_order(monkeypatch) -> None:
    thread_names: set[str] = set()
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        with lock:
            thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=1.0)
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([
        ToolSpec(name="inspect", description="Inspect an independent artifact.", can_run_parallel=True),
    ])
    runtime = ToolRuntime(registry, handler=handler)
    calls = [ToolCall(call_id=f"inspect-{index}", tool_name="inspect") for index in range(3)]
    monkeypatch.setenv("SCIDATA_AGENT_TOOL_WORKERS", "3")

    results = runtime.execute_many(calls)

    assert [result.call_id for result in results] == ["inspect-0", "inspect-1", "inspect-2"]
    assert len(thread_names) == 3


def test_tool_runtime_keeps_mixed_batches_sequential(monkeypatch) -> None:
    thread_names: list[str] = []

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        thread_names.append(call.call_id)
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([
        ToolSpec(name="inspect", description="Inspect an artifact.", can_run_parallel=True),
        ToolSpec(name="mutate", description="Update shared state.", can_run_parallel=False),
    ])
    runtime = ToolRuntime(registry, handler=handler)
    calls = [
        ToolCall(call_id="inspect-1", tool_name="inspect"),
        ToolCall(call_id="mutate-1", tool_name="mutate"),
    ]
    monkeypatch.setenv("SCIDATA_AGENT_TOOL_WORKERS", "2")

    results = runtime.execute_many(calls)

    assert [result.call_id for result in results] == ["inspect-1", "mutate-1"]
    assert thread_names == ["inspect-1", "mutate-1"]


def test_policy_filters_duplicate_calls_before_execution(tmp_path) -> None:
    state = make_state(tmp_path)
    invoked = False

    def handler(call: ToolCall, context: Any, options: dict[str, Any]) -> ToolResult:
        nonlocal invoked
        invoked = True
        return ToolResult(call_id=call.call_id, tool_name=call.tool_name, status="completed")

    registry = ToolRegistry([ToolSpec(name="search", description="Search sources.")])
    runtime = ToolRuntime(registry, handler=handler)

    def decide(observation, context) -> AgentDecision:
        return AgentDecision(
            decision="continue",
            tool_calls=[
                ToolCall(call_id="search-1", tool_name="search", arguments={"query": "same"}),
                ToolCall(call_id="search-2", tool_name="search", arguments={"query": "same"}),
            ],
        )

    result = AgentHarness(registry, decide, runtime=runtime).run(state, max_iterations=1)

    assert result.status == "partial"
    assert invoked is True
    assert any("Duplicate tool call" in reason for reason in result.stop_rejections)
    assert [item.call_id for item in result.tool_results] == ["search-1"]
    assert any(event["event_type"] == "policy_filtered" for event in result.trace)


def test_observation_bounds_context_but_preserves_catalog_counts(tmp_path, monkeypatch) -> None:
    state = make_state(tmp_path)
    state.source_catalog = [
        SourceCatalogEntry(
            source_id=f"source-{index}",
            title=f"Source {index}",
            artifacts=[
                SourceArtifact(
                    source_id=f"source-{index}",
                    name=f"artifact-{index}",
                    artifact_type="pdf",
                )
            ],
        )
        for index in range(5)
    ]
    monkeypatch.setenv("SCIDATA_AGENT_OBSERVATION_MAX_SOURCES", "2")
    monkeypatch.setenv("SCIDATA_AGENT_OBSERVATION_MAX_ARTIFACTS", "3")

    observation = ObservationBuilder().build(
        state,
        ToolRegistry([ToolSpec(name="search", description="Search sources.")]),
    )

    assert observation.sources["catalog_count"] == 5
    assert len(observation.sources["items"]) == 2
    assert observation.artifacts["total"] == 5
    assert len(observation.artifacts["items"]) == 3


def test_observation_includes_connector_failures_for_replanning(tmp_path) -> None:
    state = make_state(tmp_path)
    state.connector_status = [{
        "connector": "arxiv",
        "query": "all:example",
        "status": "failed",
        "error": "HTTP 503",
    }]

    observation = ObservationBuilder().build(
        state,
        ToolRegistry([ToolSpec(name="search_more", description="Search another source.")]),
    )

    assert any("arxiv" in failure and "HTTP 503" in failure for failure in observation.failures)


def test_observation_restores_recent_tool_results_from_state(tmp_path) -> None:
    state = make_state(tmp_path)
    state.tool_result_history = [{
        "call_id": "call-1",
        "tool_name": "search_more",
        "status": "partial",
        "data": {"new_sources": 4},
    }]

    observation = ObservationBuilder().build(
        state,
        ToolRegistry([ToolSpec(name="search_more", description="Search another source.")]),
    )

    assert len(observation.recent_results) == 1
    assert observation.recent_results[0]["call_id"] == "call-1"
    assert observation.recent_results[0]["data"] == {"new_sources": 4}
    assert "artifact_refs" in observation.recent_results[0]
