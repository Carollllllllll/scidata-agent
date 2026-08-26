from __future__ import annotations

from pathlib import Path

from scidata_agent.agent.schemas import (
    AgentState,
    DynamicExtractionPlan,
    DynamicRecord,
    InformationNeed,
    SourceArtifact,
    SourceCatalogEntry,
    TaskPlan,
)
from scidata_agent.tools.coverage import build_coverage_report


def make_state(tmp_path: Path) -> AgentState:
    return AgentState(
        research_question="Compare the requested experiments.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Compare the requested experiments.",
            source_requirements=["papers", "tables", "supplementary_materials"],
            information_needs=[
                InformationNeed(need_name="model architecture", priority="high"),
                InformationNeed(need_name="evaluation metrics", priority="medium"),
            ],
        ),
    )


def test_coverage_requires_missing_fields_and_unprocessed_relevant_artifacts(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Primary paper",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-primary",
                    source_id="source-1",
                    artifact_type="pdf",
                    relevance_score=3.8,
                    status="discovered",
                )
            ],
        )
    ]

    report = build_coverage_report(state)

    assert report.decision == "continue"
    assert report.missing_requirements == ["model architecture", "evaluation metrics"]
    assert report.required_evidence_types == [
        "paper_full_text",
        "table",
        "supplementary_material",
    ]
    assert report.unprocessed_relevant_artifacts == ["artifact-primary"]
    assert "download_artifact" in report.recommended_actions
    requirement_gap = next(
        gap for gap in report.gaps if gap.requirement_name == "model architecture"
    )
    assert requirement_gap.priority == "high"
    assert requirement_gap.missing_fields == ["model architecture"]
    assert requirement_gap.evidence_count == 0
    assert "parse_pdf_sections" in requirement_gap.recommended_actions
    assert any(gap.gap_id == "unprocessed_relevant_artifacts" for gap in report.gaps)


def test_coverage_accepts_normalized_dynamic_field_names(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.dynamic_records = [
        DynamicRecord(
            table_name="comparison",
            fields={"modelArchitecture": "Transformer", "metric_results": "0.81"},
            source_file="paper.pdf",
        )
    ]
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Primary paper",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-paper",
                    source_id="source-1",
                    artifact_type="pdf",
                    status="parsed",
                ),
                SourceArtifact(
                    artifact_id="artifact-table",
                    source_id="source-1",
                    artifact_type="csv",
                    status="parsed",
                ),
                SourceArtifact(
                    artifact_id="artifact-supplement",
                    source_id="source-1",
                    artifact_type="supplementary_pdf",
                    status="parsed",
                ),
            ],
        )
    ]

    report = build_coverage_report(state)

    assert report.decision == "allow_stop"
    assert report.missing_requirements == []
    assert report.covered_evidence_types == [
        "paper_full_text",
        "table",
        "supplementary_material",
    ]
    assert report.requirements[0].evidence_count == 1
    assert report.requirements[0].status == "covered"


def test_unknown_source_requirement_does_not_create_permanent_missing_evidence(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract the requested result.",
        files=[],
        output_dir=tmp_path / "outputs",
        task_plan=TaskPlan(target_fields=["result"]),
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract the requested result.",
            source_requirements=["custom evidence bundle"],
        ),
        dynamic_records=[
            DynamicRecord(
                table_name="results",
                fields={"result": "0.9"},
                source_file="paper.pdf",
            )
        ],
    )

    report = build_coverage_report(state)

    assert report.required_evidence_types == []
    assert report.decision == "allow_stop"


def test_high_relevance_artifact_blocks_stop_until_failed_or_parsed(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.dynamic_records = [
        DynamicRecord(
            table_name="comparison",
            fields={"model_architecture": "Transformer", "evaluation_metrics": "0.81"},
            source_file="paper.pdf",
        )
    ]
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-1",
            title="Primary paper",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-paper",
                    source_id="source-1",
                    artifact_type="pdf",
                    relevance_score=3.1,
                    status="metadata_read",
                ),
                SourceArtifact(
                    artifact_id="artifact-table",
                    source_id="source-1",
                    artifact_type="csv",
                    status="parsed",
                ),
                SourceArtifact(
                    artifact_id="artifact-supplement",
                    source_id="source-1",
                    artifact_type="supplementary_pdf",
                    status="parsed",
                ),
            ],
        )
    ]

    report = build_coverage_report(state)

    assert report.decision == "continue"
    assert report.unprocessed_relevant_artifacts == ["artifact-paper"]
    assert report.gaps[-1].requirement_name == "High-relevance artifacts"


def test_low_priority_gap_is_reported_but_does_not_block_stop(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract the requested result.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract the requested result.",
            information_needs=[InformationNeed(need_name="optional context", priority="low")],
        ),
        dynamic_records=[
            DynamicRecord(
                table_name="results",
                fields={"result": "0.9"},
                source_file="paper.pdf",
            )
        ],
    )

    report = build_coverage_report(state)

    assert report.decision == "allow_stop"
    assert report.gaps[0].priority == "low"
    assert report.gaps[0].status == "missing"


def test_optional_evidence_field_does_not_block_stop(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract available results.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract available results.",
            dynamic_tables=[
                {
                    "table_name": "results",
                    "priority": "high",
                    "fields": [
                        {
                            "name": "optional_context",
                            "required": False,
                            "evidence_required": True,
                        }
                    ],
                }
            ],
        ),
    )

    report = build_coverage_report(state)

    assert report.decision == "allow_stop"
    assert report.requirements[0].priority == "low"
    assert report.requirements[0].status == "missing"
    assert report.gaps[0].priority == "low"


def test_failed_required_evidence_is_marked_unavailable(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract the requested paper evidence.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract the requested paper evidence.",
            source_requirements=["papers"],
        ),
        source_catalog=[
            SourceCatalogEntry(
                source_id="source-1",
                title="Unavailable paper",
                artifacts=[
                    SourceArtifact(
                        artifact_id="artifact-failed",
                        source_id="source-1",
                        artifact_type="pdf",
                        status="failed",
                        failure_reason="HTTP timeout",
                    )
                ],
            )
        ],
    )

    report = build_coverage_report(state)

    evidence_gap = next(gap for gap in report.gaps if gap.gap_id.startswith("evidence_"))
    assert evidence_gap.status == "unavailable"
    assert report.decision == "continue"
