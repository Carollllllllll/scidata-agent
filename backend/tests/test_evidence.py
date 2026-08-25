from pathlib import Path

from scidata_agent.agent.schemas import (
    AgentState,
    DynamicRecord,
    SourceArtifact,
    SourceCatalogEntry,
    SourceType,
    SectionBlock,
    TableBlock,
    UploadedFile,
)
from scidata_agent.tools.evidence import build_evidence_traces


def _state(tmp_path: Path) -> AgentState:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")
    return AgentState(
        task_id="task_evidence",
        research_question="Extract a metric.",
        files=[UploadedFile(file_id="file_1", filename="paper.pdf", path=pdf, content_type="application/pdf")],
        output_dir=tmp_path,
        source_catalog=[
            SourceCatalogEntry(
                source_id="source_1",
                title="Paper",
                source_type="paper",
                artifacts=[SourceArtifact(
                    artifact_id="artifact_1",
                    source_id="source_1",
                    name="paper.pdf",
                    local_path=str(pdf),
                    artifact_type="pdf",
                    status="parsed",
                )],
            )
        ],
    )


def test_build_evidence_trace_resolves_section_and_table(tmp_path: Path):
    state = _state(tmp_path)
    state.parsed_sources.section_blocks.append(
        SectionBlock(
            source_file="paper.pdf",
            source_path=str(tmp_path / "paper.pdf"),
            section_id="section_results",
            section_title="Results",
            page_start=2,
            page_end=2,
            page=2,
            text="The metric was 7.8 units.",
            chunk_id="chunk_1",
        )
    )
    state.parsed_sources.tables.append(
        TableBlock(
            source_file="paper.pdf",
            source_path=str(tmp_path / "paper.pdf"),
            source_type=SourceType.PDF_TABLE,
            columns=["metric"],
            rows=[{"metric": 7.8}],
            table_id="table_1",
            page=2,
            extraction_method="table_transformer",
        )
    )
    state.dynamic_records = [
        DynamicRecord(
            record_id="record_1",
            table_name="metrics",
            fields={"metric": 7.8},
            source_file="paper.pdf",
            source_type=SourceType.PDF_TABLE,
            page=2,
            evidence_text="The metric was 7.8 units.",
            confidence=0.9,
        )
    ]

    traces = build_evidence_traces(state)

    assert len(traces) == 1
    assert traces[0].source_id == "source_1"
    assert traces[0].artifact_id == "artifact_1"
    assert traces[0].section_title == "Results"
    assert traces[0].table_id == "table_1"
    assert traces[0].evidence_type == "table"
    assert traces[0].extraction_method == "table_transformer"
    assert traces[0].locator_status == "resolved"


def test_build_evidence_trace_marks_missing_location_without_fabrication(tmp_path: Path):
    state = _state(tmp_path)
    state.dynamic_records = [
        DynamicRecord(
            record_id="record_2",
            table_name="metrics",
            fields={"metric": 7.8},
            source_file="unknown.txt",
            source_type=SourceType.UNKNOWN,
            confidence=0.4,
        )
    ]

    traces = build_evidence_traces(state)

    assert len(traces) == 1
    assert traces[0].page is None
    assert traces[0].table_id is None
    assert traces[0].figure_id is None
    assert traces[0].locator_status == "unresolved"
