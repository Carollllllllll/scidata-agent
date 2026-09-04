from __future__ import annotations

from pathlib import Path

from scidata_agent.agent.schemas import (
    AgentState,
    DynamicExtractionPlan,
    DynamicFieldSpec,
    DynamicRecord,
    DynamicTableSpec,
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
        "dataset",
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


def test_required_duplicate_field_promotes_priority(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract the dataset used by each experiment.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract the dataset used by each experiment.",
            dynamic_tables=[
                {
                    "table_name": "experiments",
                    "fields": [
                        {
                            "name": "dataset_or_object",
                            "required": False,
                            "evidence_required": True,
                        }
                    ],
                },
                {
                    "table_name": "dataset_usage",
                    "fields": [
                        {
                            "name": "dataset_or_object",
                            "required": True,
                            "evidence_required": True,
                        }
                    ],
                },
            ],
        ),
    )

    report = build_coverage_report(state)

    assert len(report.requirements) == 1
    assert report.requirements[0].priority == "high"
    assert report.decision == "continue"
    assert report.missing_requirements == ["dataset_or_object"]


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
    assert report.decision == "allow_stop"


def test_coverage_stops_recommending_search_more_after_two_attempts(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.runtime_search_more_count = 2
    state.runtime_search_more_limit = 2

    report = build_coverage_report(state)

    assert "search_more" not in report.recommended_actions
    assert all(
        "search_more" not in gap.recommended_actions
        for gap in report.gaps
    )


def test_field_group_becomes_sufficient_after_initial_search_and_three_sources(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Compare lifetime measurements.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Compare lifetime measurements.",
            dynamic_tables=[
                DynamicTableSpec(
                    table_name="stability",
                    fields=[DynamicFieldSpec(name="lifetime", required=True)],
                )
            ],
        ),
        runtime_group_initial_searches=["stability"],
        dynamic_records=[
            DynamicRecord(
                table_name="stability",
                fields={"lifetime": value},
                source_file=f"source-{index}.pdf",
            )
            for index, value in enumerate(("1 h", "2 h", "3 h"), start=1)
        ],
    )

    report = build_coverage_report(state)

    assert report.field_groups[0].status == "sufficient"
    assert report.field_groups[0].source_count == 3
    assert report.decision == "allow_stop"


def test_field_group_becomes_completed_exhausted_after_two_supplemental_searches(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Find an unavailable lifetime measurement.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Find an unavailable lifetime measurement.",
            dynamic_tables=[
                DynamicTableSpec(
                    table_name="stability",
                    fields=[DynamicFieldSpec(name="lifetime", required=True)],
                )
            ],
        ),
        runtime_group_initial_searches=["stability"],
        runtime_group_search_more_counts={"stability": 2},
        runtime_search_more_limit=2,
    )

    report = build_coverage_report(state)

    assert report.field_groups[0].status == "exhausted"
    assert report.field_groups[0].coverage_score == 0.0
    assert report.decision == "allow_stop"
    assert "search_more" not in report.recommended_actions


def test_inspected_high_relevance_artifact_still_requires_content_parsing(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.source_catalog = [
        SourceCatalogEntry(
            source_id="source-inspected",
            title="Inspected metadata source",
            artifacts=[
                SourceArtifact(
                    artifact_id="artifact-inspected",
                    source_id="source-inspected",
                    artifact_type="landing_page",
                    relevance_score=4.0,
                    status="inspected",
                    url="https://example.org/paper",
                    completed_operations=["read_metadata"],
                )
            ],
        )
    ]

    report = build_coverage_report(state)

    assert report.unprocessed_relevant_artifacts == ["artifact-inspected"]


def test_structured_file_covers_table_and_dataset_requirements(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract dataset values.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract dataset values.",
            source_requirements=["tables", "datasets"],
        ),
        source_catalog=[
            SourceCatalogEntry(
                source_id="dataset-source",
                title="Structured dataset",
                artifacts=[
                    SourceArtifact(
                        artifact_id="dataset-csv",
                        source_id="dataset-source",
                        artifact_type="csv",
                        status="parsed",
                    )
                ],
            )
        ],
    )

    report = build_coverage_report(state)

    assert report.required_evidence_types == ["table", "dataset"]
    assert report.covered_evidence_types == ["table", "dataset"]
    assert report.decision == "allow_stop"


def test_dynamic_field_coverage_uses_its_own_table_as_denominator(tmp_path: Path) -> None:
    state = AgentState(
        research_question="Extract linked measurements.",
        files=[],
        output_dir=tmp_path / "outputs",
        dynamic_extraction_plan=DynamicExtractionPlan(
            research_goal="Extract linked measurements.",
            information_needs=[
                InformationNeed(need_name="Narrative label that does not match fields", priority="high")
            ],
            dynamic_tables=[
                {
                    "table_name": "events",
                    "fields": [
                        {"name": "event_id", "required": True, "evidence_required": True}
                    ],
                },
                {
                    "table_name": "measurements",
                    "fields": [
                        {"name": "value", "required": True, "evidence_required": True}
                    ],
                },
            ],
        ),
        dynamic_records=[
            DynamicRecord(table_name="events", fields={"event_id": "SN-1"}, source_file="a.csv"),
            DynamicRecord(table_name="measurements", fields={"value": 1.2}, source_file="a.csv"),
            DynamicRecord(table_name="measurements", fields={"value": 1.3}, source_file="a.csv"),
            DynamicRecord(table_name="measurements", fields={"value": 1.4}, source_file="a.csv"),
        ],
    )

    report = build_coverage_report(state)

    assert [item.name for item in report.requirements] == ["event_id", "value"]
    assert all(item.status == "covered" for item in report.requirements)
    assert report.decision == "allow_stop"
