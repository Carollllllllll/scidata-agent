from __future__ import annotations

from pathlib import Path

import pytest

from scidata_agent.agent.action_executor import (
    ArtifactActionExecutor,
    _apply_search_strategy,
    _normalize_search_strategy,
    _tool_call_from_action,
    effective_extraction_blocks,
    next_required_derived_stage,
    parsed_content_fingerprint,
    source_content_fingerprint,
)
from scidata_agent.agent.action_preflight import preflight_artifact_action_plan
from scidata_agent.agent.action_registry import artifact_type_supported, list_action_capabilities
from scidata_agent.agent.schemas import (
    AgentState,
    ArtifactAction,
    ArtifactActionPlan,
    ArtifactActionResult,
    DiscoveredSource,
    DynamicExtractionPlan,
    MultiSourceSearchPlan,
    SourceDiscoveryPlan,
    SourceSearchRequest,
    SourceArtifact,
    SourceCatalogEntry,
    ScientificRecord,
    SectionBlock,
    TaskPlan,
    TextBlock,
)
from scidata_agent.tools import source_ingestion


def make_state(tmp_path: Path) -> AgentState:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("method,RMSE\nmodel-a,1.2\nmodel-b,0.8\n", encoding="utf-8")
    artifact = SourceArtifact(
        artifact_id="artifact_csv_1",
        source_id="source_dataset_1",
        artifact_type="csv",
        local_path=str(csv_path),
        status="downloaded",
    )
    return AgentState(
        research_question="Compare model RMSE results.",
        files=[],
        output_dir=tmp_path / "outputs",
        source_catalog=[
            SourceCatalogEntry(
                source_id="source_dataset_1",
                title="Model results dataset",
                source_type="dataset",
                provider="figshare",
                relevance_score=0.9,
                artifacts=[artifact],
                metadata={"files": [{"name": "results.csv", "size": 52}]},
            )
        ],
    )


def action(action_name: str, artifact_id: str | None = "artifact_csv_1") -> ArtifactAction:
    return ArtifactAction(
        action_id=f"action_{action_name}",
        artifact_id=artifact_id,
        action=action_name,
        purpose="test action",
        reason="test reason",
    )


def test_registry_exposes_supported_actions_and_types() -> None:
    capabilities = {item.action: item for item in list_action_capabilities()}

    assert "parse_pdf_sections" in capabilities
    assert "parse_figure" in capabilities
    assert artifact_type_supported("parse_csv", "csv")
    assert not artifact_type_supported("parse_pdf_text", "csv")
    assert artifact_type_supported("read_metadata", "csv")


def test_workflow_global_action_is_routed_to_handler(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    seen: list[str] = []

    def workflow_handler(selected_action: ArtifactAction, selected_state: AgentState) -> ArtifactActionResult:
        seen.append(selected_action.action)
        return ArtifactActionResult(
            action_id=selected_action.action_id,
            artifact_id=None,
            action=selected_action.action,
            status="completed",
            message="workflow stage executed",
        )

    executor = ArtifactActionExecutor(workflow_handler=workflow_handler)
    result = executor.execute_action(
        ArtifactAction(
            action_id="select-1",
            artifact_id=None,
            action="select_sources",
            purpose="select sources",
            reason="test workflow routing",
        ),
        state,
    )

    assert result.status == "completed"
    assert seen == ["select_sources"]


def test_executor_reads_csv_into_shared_state(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    plan = ArtifactActionPlan(
        research_goal=state.research_question,
        actions=[action("parse_csv")],
    )

    results = ArtifactActionExecutor().execute_plan(plan, state)

    assert len(results) == 1
    assert results[0].status == "completed"
    assert results[0].output_counts == {"tables": 1, "rows": 2}
    assert state.parsed_sources.tables[0].columns == ["method", "RMSE"]
    assert state.parsed_sources.tables[0].rows[1]["RMSE"] == 0.8
    assert any("Artifact action action_parse_csv" in line for line in state.processing_log)


def test_executor_skips_duplicate_parse_for_already_parsed_artifact(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    plan = ArtifactActionPlan(
        research_goal=state.research_question,
        actions=[action("parse_csv"), action("parse_csv")],
    )

    results = ArtifactActionExecutor().execute_plan(plan, state)

    assert [result.status for result in results] == ["completed", "skipped"]
    assert len(state.parsed_sources.tables) == 1
    assert state.source_catalog[0].artifacts[0].status == "parsed"


def test_executor_records_metadata_without_downloading(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    results = ArtifactActionExecutor().execute_plan(
        ArtifactActionPlan(
            research_goal=state.research_question,
            actions=[action("read_metadata")],
        ),
        state,
    )

    assert results[0].status == "completed"
    assert results[0].output_counts["metadata_fields"] > 0
    assert state.source_insights[-1].insight_type == "metadata"


def test_executor_downloads_selected_remote_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = SourceArtifact(
        artifact_id="artifact_remote_pdf",
        source_id="source_remote",
        artifact_type="pdf",
        name="paper.pdf",
        url="https://example.org/paper.pdf",
        status="planned",
    )
    state = AgentState(
        research_question="Read the paper.",
        files=[],
        output_dir=tmp_path / "outputs",
        source_catalog=[
            SourceCatalogEntry(
                source_id="source_remote",
                title="Remote paper",
                source_type="paper",
                artifacts=[artifact],
            )
        ],
    )

    def fake_download(url: str, target: Path, max_bytes: int) -> None:
        assert url.endswith("paper.pdf")
        assert max_bytes > 0
        target.write_bytes(b"%PDF-1.7 test")

    monkeypatch.setattr(source_ingestion, "_download_url", fake_download)
    result = ArtifactActionExecutor().execute_action(
        ArtifactAction(
            action_id="action_download",
            artifact_id=artifact.artifact_id,
            action="download_artifact",
            purpose="Read the paper content.",
            reason="The paper is relevant.",
        ),
        state,
    )

    assert result.status == "completed"
    assert artifact.status == "downloaded"
    assert artifact.local_path and Path(artifact.local_path).exists()
    assert artifact.size_bytes == len(b"%PDF-1.7 test")


def test_executor_download_artifact_preserves_missing_url_failure(tmp_path: Path) -> None:
    artifact = SourceArtifact(
        artifact_id="artifact_missing_url",
        source_id="source_missing",
        artifact_type="csv",
        name="results.csv",
        status="planned",
    )
    state = AgentState(
        research_question="Read the data.",
        files=[],
        output_dir=tmp_path / "outputs",
        source_catalog=[SourceCatalogEntry(source_id="source_missing", title="Missing", artifacts=[artifact])],
    )

    result = ArtifactActionExecutor().execute_action(
        ArtifactAction(
            action_id="action_missing_download",
            artifact_id=artifact.artifact_id,
            action="download_artifact",
            purpose="Read the data.",
            reason="The selected data file is relevant.",
        ),
        state,
    )

    assert result.status == "failed"
    assert "no URL" in result.message
    assert artifact.status == "failed"
    assert artifact.failure_reason


def test_source_artifact_download_rejects_invalid_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeResponse:
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self, _limit: int) -> bytes:
            return b"<html>not a PDF</html>"

    monkeypatch.setattr(source_ingestion, "safe_urlopen", lambda request, timeout: FakeResponse())
    artifact = SourceArtifact(
        artifact_id="artifact_invalid_pdf",
        source_id="source_invalid",
        artifact_type="pdf",
        name="paper.pdf",
        url="https://example.org/paper.pdf?download=1",
    )

    with pytest.raises(RuntimeError, match="not a PDF"):
        source_ingestion.download_source_artifact(artifact, tmp_path / "downloads")

    assert artifact.local_path is None
    assert artifact.status == "discovered"


def test_executor_does_not_use_wrong_parser_for_artifact_type(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    result = ArtifactActionExecutor().execute_action(action("parse_pdf_text"), state)

    assert result.status == "skipped"
    assert "does not support" in result.message
    assert state.parsed_sources.text_blocks == []


def test_executor_preserves_missing_local_file_as_failure(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_catalog[0].artifacts[0].local_path = str(tmp_path / "missing.csv")
    result = ArtifactActionExecutor().execute_action(action("parse_csv"), state)

    assert result.status == "failed"
    assert "does not exist" in result.message


def test_global_actions_are_recorded_without_fake_scientific_output(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    plan = ArtifactActionPlan(
        research_goal=state.research_question,
        should_continue=False,
        actions=[action("search_more", artifact_id=None), action("stop", artifact_id=None)],
    )

    results = ArtifactActionExecutor().execute_plan(plan, state)

    assert [result.status for result in results] == ["failed", "no_op"]
    assert state.parsed_sources.tables == []
    assert state.parsed_sources.text_blocks == []


class SearchMorePlanner:
    def plan_multi_source_search(self, research_question: str, source_discovery_plan: SourceDiscoveryPlan) -> MultiSourceSearchPlan:
        return MultiSourceSearchPlan(
            research_goal=research_question,
            search_requests=[
                SourceSearchRequest(
                    connector_name="crossref",
                    source_type="paper_metadata",
                    query="model RMSE results",
                    purpose="Find additional bibliographic evidence.",
                )
            ],
        )


def test_search_more_runs_a_new_llm_planned_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[],
    )

    def fake_search(plan: MultiSourceSearchPlan):
        return (
            [
                DiscoveredSource(
                    title="Additional RMSE paper",
                    source_type="paper_metadata",
                    url="https://example.org/rmse-paper",
                    metadata={"provider": "crossref"},
                )
            ],
            {
                "status": "completed",
                "searched": 1,
                "failed": 0,
                "connector_status": [{"connector": "crossref", "status": "completed", "added": 1}],
            },
        )

    monkeypatch.setattr(
        "scidata_agent.agent.action_executor.execute_multi_source_search",
        fake_search,
    )
    result = ArtifactActionExecutor(SearchMorePlanner()).execute_action(
        action("search_more", artifact_id=None),
        state,
    )

    assert result.status == "completed"
    assert result.output_counts == {
        "search_requests": 1,
        "new_sources": 1,
        "failed_requests": 0,
        "planning_batches": 1,
        "failed_planning_batches": 0,
    }
    assert state.multi_source_search_plan is not None
    assert len(state.source_discovery_plan.candidate_sources) == 1


def test_search_more_respects_the_per_task_limit(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[],
    )
    state.runtime_search_more_count = 2
    state.runtime_search_more_limit = 2

    result = ArtifactActionExecutor(SearchMorePlanner()).execute_action(
        action("search_more", artifact_id=None),
        state,
    )

    assert result.status == "skipped"
    assert "limit exhausted" in result.message
    assert state.runtime_search_more_count == 2


def test_dynamic_extraction_idempotency_changes_when_parsed_content_changes(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    extraction = action("extract_dynamic_records", artifact_id=None)
    first = _tool_call_from_action(
        extraction,
        parsed_content_fingerprint=parsed_content_fingerprint(state),
    )
    state.parsed_sources.text_blocks.append(
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            source_type="pdf_text",
            page=1,
            text="SN 2011fe peak magnitude -19.3",
            chunk_id="chunk-1",
        )
    )
    second = _tool_call_from_action(
        extraction,
        parsed_content_fingerprint=parsed_content_fingerprint(state),
    )

    assert first.effective_idempotency_key() != second.effective_idempotency_key()


def test_extraction_batch_changes_when_parsed_artifact_becomes_high_relevance(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.parsed_sources.text_blocks.append(
        TextBlock(
            source_file="results.csv",
            source_path=str(tmp_path / "results.csv"),
            source_type="csv",
            text="model-a RMSE 1.2",
            chunk_id="chunk-results",
        )
    )
    first = parsed_content_fingerprint(state)

    state.source_catalog[0].artifacts[0].relevance_score = 3.0

    assert parsed_content_fingerprint(state) != first


@pytest.mark.parametrize(
    "stage",
    [
        "extract_figures",
        "interpret_sections",
        "extract_dynamic_records",
        "extract_records",
        "normalize_records",
        "track_provenance",
        "validate_quality",
    ],
)
def test_every_derived_stage_is_scoped_to_the_extraction_batch(
    stage: str,
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    selected = action(stage, artifact_id=None)
    first = _tool_call_from_action(selected, parsed_content_fingerprint="batch-a")
    second = _tool_call_from_action(selected, parsed_content_fingerprint="batch-b")

    assert first.effective_idempotency_key() != second.effective_idempotency_key()


def test_effective_extraction_blocks_keep_raw_text_for_sources_without_sections(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.parsed_sources.section_blocks.append(
        SectionBlock(
            source_file="paper-a.pdf",
            source_path="paper-a.pdf",
            text="section-aware A",
            chunk_id="section-a",
        )
    )
    state.parsed_sources.text_blocks.extend(
        [
            TextBlock(
                source_file="paper-a.pdf",
                source_path="paper-a.pdf",
                text="raw duplicate A",
                chunk_id="raw-a",
            ),
            TextBlock(
                source_file="paper-b.pdf",
                source_path="paper-b.pdf",
                text="new raw B",
                chunk_id="raw-b",
            ),
        ]
    )

    blocks = effective_extraction_blocks(state)

    assert [block.chunk_id for block in blocks] == ["section-a", "raw-b"]


def test_preflight_removes_search_more_after_limit_is_exhausted(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.runtime_search_more_count = 2
    state.runtime_search_more_limit = 2
    plan = ArtifactActionPlan(
        research_goal=state.research_question,
        iteration=3,
        actions=[action("search_more", artifact_id=None)],
    )

    dropped = preflight_artifact_action_plan(plan, state)

    assert not plan.actions
    assert dropped and "limit is exhausted" in dropped[0]


def test_new_extraction_batch_reopens_the_complete_derived_stage_chain(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_catalog[0].artifacts[0].status = "parsed"
    state.task_plan = TaskPlan(research_goal=state.research_question)
    state.dynamic_extraction_plan = DynamicExtractionPlan(research_goal=state.research_question)
    state.parsed_sources.text_blocks.append(
        TextBlock(
            source_file="results.csv",
            source_path=str(tmp_path / "results.csv"),
            source_type="csv",
            text="model-a RMSE 1.2",
            chunk_id="chunk-results",
        )
    )
    source_fingerprint = source_content_fingerprint(state)
    fingerprint = parsed_content_fingerprint(state)

    expected_stages = [
        "extract_figures",
        "interpret_sections",
        "extract_dynamic_records",
        "extract_records",
        "normalize_records",
        "track_provenance",
        "validate_quality",
    ]
    for stage in expected_stages:
        assert next_required_derived_stage(state) == stage
        state.runtime_stage_fingerprints[stage] = (
            source_fingerprint
            if stage in {"extract_figures", "interpret_sections"}
            else fingerprint
        )

    assert next_required_derived_stage(state) is None

    state.source_catalog[0].artifacts[0].relevance_score = 3.0

    assert next_required_derived_stage(state) == "extract_figures"


def test_figure_and_section_stages_use_stable_preprocessing_fingerprint(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_catalog[0].artifacts[0].status = "parsed"
    state.parsed_sources.text_blocks.append(
        TextBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            text="Results and discussion",
            chunk_id="raw-paper",
        )
    )
    fingerprint = source_content_fingerprint(state)

    state.runtime_stage_fingerprints["extract_figures"] = fingerprint
    state.parsed_sources.section_blocks.append(
        SectionBlock(
            source_file="paper.pdf",
            source_path="paper.pdf",
            text="Results and discussion",
            chunk_id="section-paper",
        )
    )

    assert source_content_fingerprint(state) == fingerprint
    assert next_required_derived_stage(state) == "interpret_sections"


def test_preprocessing_runs_for_available_content_while_an_artifact_is_pending(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.parsed_sources.text_blocks.append(
        TextBlock(
            source_file="available.pdf",
            source_path="available.pdf",
            text="Available evidence",
            chunk_id="available",
        )
    )
    state.coverage_report.unprocessed_relevant_artifacts = ["artifact-pending"]

    assert next_required_derived_stage(state) == "extract_figures"


def test_search_more_propagates_partial_connector_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[],
    )

    def fake_search(plan: MultiSourceSearchPlan):
        return (
            [
                DiscoveredSource(
                    title="Partial fallback paper",
                    source_type="paper_metadata",
                    url="https://example.org/partial-paper",
                    metadata={"provider": "openalex"},
                )
            ],
            {
                "status": "partial",
                "searched": len(plan.search_requests),
                "failed": 1,
                "connector_status": [
                    {"connector": "arxiv", "status": "failed", "error": "timeout"},
                    {"connector": "openalex", "status": "completed", "added": 1},
                ],
            },
        )

    monkeypatch.setattr(
        "scidata_agent.agent.action_executor.execute_multi_source_search",
        fake_search,
    )
    result = ArtifactActionExecutor(SearchMorePlanner()).execute_action(
        action("search_more", artifact_id=None),
        state,
    )

    assert result.status == "partial"
    assert result.output_counts["failed_requests"] == 1
    assert "status=partial" in result.message


def test_search_more_uses_bounded_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                title=f"Candidate {index}",
                source_type="paper_metadata",
                url=f"https://example.org/{index}",
                metadata={"provider": "crossref"},
            )
            for index in range(85)
        ],
    )
    calls: list[tuple[int, int, str]] = []

    class BoundedPlanner:
        def plan_multi_source_search(
            self,
            research_question: str,
            source_discovery_plan: SourceDiscoveryPlan,
            *,
            candidate_context_limit: int = 40,
            batch_label: str = "single batch",
        ) -> MultiSourceSearchPlan:
            calls.append((len(source_discovery_plan.candidate_sources), candidate_context_limit, batch_label))
            return MultiSourceSearchPlan(
                research_goal=research_question,
                search_requests=[
                    SourceSearchRequest(
                        connector_name="crossref",
                        source_type="paper_metadata",
                        query=f"query {batch_label}",
                    )
                ],
            )

    def fake_search(plan: MultiSourceSearchPlan):
        return [], {"status": "completed", "searched": len(plan.search_requests), "failed": 0, "connector_status": []}

    monkeypatch.setenv("SCIDATA_SEARCH_MORE_BATCH_SIZE", "40")
    monkeypatch.setattr("scidata_agent.agent.action_executor.execute_multi_source_search", fake_search)
    result = ArtifactActionExecutor(BoundedPlanner()).execute_action(
        action("search_more", artifact_id=None), state
    )

    assert result.status == "completed"
    assert calls == [(40, 40, "batch 1/3"), (40, 40, "batch 2/3"), (5, 40, "batch 3/3")]
    assert result.output_counts["planning_batches"] == 3
    assert result.output_counts["failed_planning_batches"] == 0


def test_search_more_caps_large_catalog_before_batch_planning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                title=f"Candidate {index}",
                source_type="paper_metadata",
                url=f"https://example.org/{index}",
                metadata={"provider": "crossref"},
            )
            for index in range(160)
        ],
    )
    calls: list[tuple[int, int, str]] = []

    class BoundedPlanner:
        def plan_multi_source_search(
            self,
            research_question: str,
            source_discovery_plan: SourceDiscoveryPlan,
            *,
            candidate_context_limit: int = 40,
            batch_label: str = "single batch",
        ) -> MultiSourceSearchPlan:
            calls.append((len(source_discovery_plan.candidate_sources), candidate_context_limit, batch_label))
            return MultiSourceSearchPlan(
                research_goal=research_question,
                search_requests=[
                    SourceSearchRequest(
                        connector_name="crossref",
                        source_type="paper_metadata",
                        query=f"query {batch_label}",
                    )
                ],
            )

    monkeypatch.setenv("SCIDATA_SEARCH_MORE_BATCH_SIZE", "40")
    monkeypatch.setenv("SCIDATA_SEARCH_MORE_CANDIDATE_LIMIT", "999")
    monkeypatch.setenv("SCIDATA_SEARCH_MORE_MAX_PLANNING_BATCHES", "999")
    monkeypatch.setattr(
        "scidata_agent.agent.action_executor.execute_multi_source_search",
        lambda plan: ([], {"status": "completed", "searched": 0, "failed": 0, "connector_status": []}),
    )

    result = ArtifactActionExecutor(BoundedPlanner()).execute_action(
        action("search_more", artifact_id=None), state
    )

    assert result.status == "completed"
    assert calls == [(40, 40, "batch 1/3"), (40, 40, "batch 2/3"), (20, 40, "batch 3/3")]
    assert result.output_counts["planning_candidates"] == 100
    assert result.output_counts["planning_batches"] == 3


def test_search_more_supports_legacy_two_argument_planner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                title="Candidate",
                source_type="paper_metadata",
                url="https://example.org/candidate",
                metadata={"provider": "crossref"},
            )
        ],
    )

    def fake_search(plan: MultiSourceSearchPlan):
        return [], {"status": "completed", "searched": 1, "failed": 0, "connector_status": []}

    monkeypatch.setattr("scidata_agent.agent.action_executor.execute_multi_source_search", fake_search)
    result = ArtifactActionExecutor(SearchMorePlanner()).execute_action(
        action("search_more", artifact_id=None), state
    )

    assert result.status == "completed"
    assert result.output_counts["planning_batches"] == 1


def test_search_more_keeps_successful_batches_when_one_planning_batch_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                title=f"Candidate {index}",
                source_type="paper_metadata",
                url=f"https://example.org/{index}",
                metadata={"provider": "crossref"},
            )
            for index in range(85)
        ],
    )

    class PartiallyFailingPlanner:
        def plan_multi_source_search(
            self,
            research_question: str,
            source_discovery_plan: SourceDiscoveryPlan,
            *,
            batch_label: str = "single batch",
            **kwargs,
        ) -> MultiSourceSearchPlan:
            if batch_label == "batch 2/3":
                raise RuntimeError("temporary planner failure")
            return MultiSourceSearchPlan(
                research_goal=research_question,
                search_requests=[
                    SourceSearchRequest(
                        connector_name="crossref",
                        source_type="paper_metadata",
                        query=f"query {batch_label}",
                    )
                ],
            )

    def fake_search(plan: MultiSourceSearchPlan):
        return [], {"status": "completed", "searched": len(plan.search_requests), "failed": 0, "connector_status": []}

    monkeypatch.setenv("SCIDATA_SEARCH_MORE_BATCH_SIZE", "40")
    monkeypatch.setattr("scidata_agent.agent.action_executor.execute_multi_source_search", fake_search)
    result = ArtifactActionExecutor(PartiallyFailingPlanner()).execute_action(
        action("search_more", artifact_id=None), state
    )

    assert result.status == "completed"
    assert result.output_counts["planning_batches"] == 3
    assert result.output_counts["failed_planning_batches"] == 1
    assert result.output_counts["search_requests"] == 2
    assert any("temporary planner failure" in line for line in state.processing_log)


def test_search_more_fails_when_all_planning_batches_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    state.source_discovery_plan = SourceDiscoveryPlan(
        research_goal=state.research_question,
        candidate_sources=[
            DiscoveredSource(
                title=f"Candidate {index}",
                source_type="paper_metadata",
                url=f"https://example.org/{index}",
                metadata={"provider": "crossref"},
            )
            for index in range(2)
        ],
    )

    class FailingPlanner:
        def plan_multi_source_search(self, research_question: str, source_discovery_plan: SourceDiscoveryPlan, **kwargs):
            raise RuntimeError("planner unavailable")

    result = ArtifactActionExecutor(FailingPlanner()).execute_action(
        action("search_more", artifact_id=None), state
    )

    assert result.status == "failed"
    assert "All search_more planning batches failed" in result.message


def test_search_recovery_strategy_switches_connector_and_revises_query() -> None:
    plan = MultiSourceSearchPlan(
        research_goal="Compare model RMSE results.",
        search_requests=[
            SourceSearchRequest(
                connector_name="arxiv",
                source_type="paper_metadata",
                query="failed broad query",
            ),
            SourceSearchRequest(
                connector_name="openalex",
                source_type="paper_metadata",
                query="old openalex query",
            ),
        ],
    )
    strategy = _normalize_search_strategy(
        {
            "connector_names": ["OpenAlex", "not-a-connector"],
            "avoid_connectors": ["arxiv"],
            "source_types": ["paper_metadata"],
            "query_focus": "benchmark RMSE tables",
            "revised_queries": {"OPENALEX": ["benchmark RMSE table reproducibility"]},
            "failure_reason": "arXiv returned HTTP 503",
        }
    )

    recovered = _apply_search_strategy(plan, strategy)

    assert [request.connector_name for request in recovered.search_requests] == ["openalex"]
    assert [request.query for request in recovered.search_requests] == ["benchmark RMSE table reproducibility"]
    assert strategy["failure_reason"] == "arXiv returned HTTP 503"


class EvidenceValidator:
    def validate_records(self, records: list[ScientificRecord]):
        return []


def test_validate_evidence_updates_quality_report(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.final_records = [
        ScientificRecord(
            metric_name="RMSE",
            metric_value=0.8,
            unit="dimensionless",
            source_file="results.csv",
            evidence_text="model-b RMSE=0.8",
        )
    ]

    result = ArtifactActionExecutor(EvidenceValidator()).execute_action(
        action("validate_evidence", artifact_id=None),
        state,
    )

    assert result.status == "completed"
    assert result.output_counts["records"] == 1
    assert state.quality_report.record_count == 1
