from pathlib import Path

from scidata_agent.agent.schemas import (
    AgentState,
    DiscoveredSource,
    SourceDiscoveryPlan,
    SourceInsight,
    SourceSelectionDecision,
    SourceSelectionPlan,
    UploadedFile,
)
from scidata_agent.tools.source_catalog import (
    build_source_catalog,
    refresh_source_catalog,
    source_catalog_rows,
    source_catalog_summary,
)
from scidata_agent.tools.connectors.registry import merge_sources


def _state(tmp_path: Path, sources: list[DiscoveredSource], **kwargs) -> AgentState:
    return AgentState(
        research_question="catalog test",
        files=kwargs.pop("files", []),
        output_dir=tmp_path / "outputs",
        source_discovery_plan=SourceDiscoveryPlan(
            research_goal="catalog test",
            candidate_sources=sources,
        ),
        **kwargs,
    )


def test_pdf_source_catalog_tracks_downloaded_artifact(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    source = DiscoveredSource(
        title="Demo paper",
        source_type="paper",
        url="https://arxiv.org/abs/1234.5678",
        metadata={
            "provider": "arxiv",
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
            "downloaded_path": str(pdf_path),
        },
    )

    catalog = build_source_catalog(_state(tmp_path, [source]))

    entry = catalog[0]
    pdf = next(artifact for artifact in entry.artifacts if artifact.artifact_type == "pdf")
    assert entry.status == "downloaded"
    assert pdf.status == "downloaded"
    assert pdf.local_path == str(pdf_path)


def test_github_readme_is_a_parsed_artifact(tmp_path: Path) -> None:
    source = DiscoveredSource(
        title="demo/research-code",
        source_type="repository",
        url="https://github.com/demo/research-code",
        metadata={"provider": "github"},
    )
    state = _state(
        tmp_path,
        [source],
        source_insights=[
            SourceInsight(
                source_id=source.source_id,
                title=source.title,
                provider="github",
                source_type="repository",
                insight_type="readme",
                content="# Demo code",
                url="https://raw.githubusercontent.com/demo/research-code/main/README.md",
            )
        ],
    )

    entry = build_source_catalog(state)[0]

    readme = next(artifact for artifact in entry.artifacts if artifact.artifact_type == "readme")
    assert entry.status == "parsed"
    assert readme.status == "parsed"
    assert readme.parser == "readme_text"


def test_download_error_is_preserved_as_failure(tmp_path: Path) -> None:
    source = DiscoveredSource(
        title="Unavailable paper",
        source_type="paper",
        url="https://example.org/paper",
        metadata={"provider": "example", "pdf_url": "https://example.org/paper.pdf"},
    )
    state = _state(
        tmp_path,
        [source],
        source_insights=[
            SourceInsight(
                source_id=source.source_id,
                title=source.title,
                provider="example",
                source_type="paper",
                insight_type="download_error",
                content="HTTP 503",
                url="https://example.org/paper.pdf",
            )
        ],
    )

    entry = build_source_catalog(state)[0]

    assert entry.status == "failed"
    assert entry.failure_reason == "HTTP 503"
    assert any(artifact.status == "failed" for artifact in entry.artifacts)


def test_rejected_source_and_uploaded_file_are_explicit(tmp_path: Path) -> None:
    rejected = DiscoveredSource(
        title="Rejected source",
        source_type="webpage",
        url="https://example.org/rejected",
    )
    uploaded_path = tmp_path / "uploaded.csv"
    uploaded_path.write_text("metric,value\naccuracy,0.9\n", encoding="utf-8")
    state = _state(
        tmp_path,
        [rejected],
        source_selection_plan=SourceSelectionPlan(
            research_goal="catalog test",
            decisions=[
                SourceSelectionDecision(
                    source_id=rejected.source_id,
                    decision="reject",
                    reason="Off topic",
                )
            ],
        ),
        files=[UploadedFile(filename=uploaded_path.name, path=uploaded_path)],
    )

    catalog = build_source_catalog(state)
    rejected_entry = next(entry for entry in catalog if entry.source_id == rejected.source_id)
    uploaded_entry = next(entry for entry in catalog if entry.source_type == "uploaded_file")

    assert rejected_entry.status == "skipped"
    assert rejected_entry.artifacts[0].status == "skipped"
    assert uploaded_entry.artifacts[0].artifact_type == "csv"
    assert uploaded_entry.artifacts[0].status == "downloaded"


def test_catalog_rows_flatten_one_row_per_artifact(tmp_path: Path) -> None:
    source = DiscoveredSource(
        title="Dataset",
        source_type="dataset",
        url="https://example.org/dataset",
        metadata={
            "provider": "example",
            "files": [
                {"name": "results.csv", "download_url": "https://example.org/results.csv"},
                {"name": "readme.md", "download_url": "https://example.org/readme.md"},
            ],
        },
    )

    catalog = build_source_catalog(_state(tmp_path, [source]))
    rows = source_catalog_rows(catalog)

    assert len(rows) == len(catalog[0].artifacts)
    assert {row["artifact_type"] for row in rows} >= {"landing_page", "csv", "readme"}
    csv_row = next(row for row in rows if row["artifact_type"] == "csv")
    assert "artifact_provider" in csv_row
    assert "artifact_name" in csv_row
    assert "artifact_size_bytes" in csv_row


def test_clustered_provider_manifests_become_one_provenance_rich_artifact(tmp_path: Path) -> None:
    attachment_url = "https://data.example.org/results.csv"
    first = DiscoveredSource(
        title="Shared results",
        source_type="paper_metadata",
        url="https://doi.org/10.1234/shared-results",
        metadata={
            "provider": "crossref",
            "doi": "10.1234/shared-results",
            "files": [{"name": "results.csv", "download_url": attachment_url, "size": 128}],
        },
    )
    second = DiscoveredSource(
        title="Shared results",
        source_type="paper_metadata",
        url="https://openalex.org/W1",
        metadata={
            "provider": "openalex",
            "doi": "10.1234/shared-results",
            "files": [{"filename": "table.csv", "url": attachment_url, "size_bytes": 128}],
        },
    )
    merged, _ = merge_sources([first], [second])
    catalog = build_source_catalog(_state(tmp_path, merged))
    artifacts = [artifact for artifact in catalog[0].artifacts if artifact.url == attachment_url]

    assert len(merged) == 1
    assert len(artifacts) == 1
    assert artifacts[0].source_cluster_id == merged[0].source_cluster_id
    assert artifacts[0].provider == "crossref"
    assert artifacts[0].name == "results.csv"
    assert artifacts[0].size_bytes == 128


def test_refresh_catalog_updates_agent_state_and_reports_statuses(tmp_path: Path) -> None:
    source = DiscoveredSource(
        title="Refreshable source",
        source_type="paper",
        url="https://example.org/paper",
        metadata={"provider": "example", "pdf_url": "https://example.org/paper.pdf"},
    )
    state = _state(tmp_path, [source])

    first_catalog = refresh_source_catalog(state)
    assert state.source_catalog == first_catalog
    assert source_catalog_summary(first_catalog)["source_catalog_count"] == 1

    downloaded = tmp_path / "paper.pdf"
    downloaded.write_bytes(b"%PDF-test")
    source.metadata["downloaded_path"] = str(downloaded)
    refreshed = refresh_source_catalog(state)
    summary = source_catalog_summary(refreshed)

    assert state.source_catalog == refreshed
    assert refreshed[0].status == "downloaded"
    assert summary["source_artifacts_count"] == 2
    assert summary["source_artifact_statuses"]["downloaded"] == 1
