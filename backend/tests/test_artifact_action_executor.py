from __future__ import annotations

from pathlib import Path

import pytest

from scidata_agent.agent.action_executor import ArtifactActionExecutor
from scidata_agent.agent.action_registry import artifact_type_supported, list_action_capabilities
from scidata_agent.agent.schemas import (
    AgentState,
    ArtifactAction,
    ArtifactActionPlan,
    DiscoveredSource,
    MultiSourceSearchPlan,
    SourceDiscoveryPlan,
    SourceSearchRequest,
    SourceArtifact,
    SourceCatalogEntry,
    ScientificRecord,
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
    assert result.output_counts == {"search_requests": 1, "new_sources": 1, "failed_requests": 0}
    assert state.multi_source_search_plan is not None
    assert len(state.source_discovery_plan.candidate_sources) == 1


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
