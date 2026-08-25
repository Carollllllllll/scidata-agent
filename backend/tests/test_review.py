from __future__ import annotations

from scidata_agent.agent.schemas import (
    AgentState,
    ChartValidationIssue,
    ChartValidationResult,
    ConflictIssue,
    CoverageGap,
    CrossModalCheck,
    DynamicRecord,
    EvidenceTrace,
)
from scidata_agent.tools.review import build_review_queue


def test_review_queue_unifies_audit_risks_without_mutating_extraction() -> None:
    record = DynamicRecord(
        record_id="record-1",
        table_name="results",
        fields={"score": 0.8},
        source_file="paper.pdf",
        warnings=["low confidence"],
        raw={"original_value": 0.8},
    )
    state = AgentState(research_question="compare methods", files=[], output_dir="outputs")
    state.needs_review_records = [record]
    state.evidence_traces = [
        EvidenceTrace(
            evidence_id="evidence-1",
            record_id="record-1",
            source_file="paper.pdf",
            evidence_text="score 0.8",
        )
    ]
    state.chart_validations = [
        ChartValidationResult(
            figure_id="figure-1",
            passed=False,
            needs_review=True,
            issues=[ChartValidationIssue(code="low_confidence", message="Chart values are approximate.")],
        )
    ]
    state.cross_modal_checks = [
        CrossModalCheck(
            check_id="check-1",
            source_file="paper.pdf",
            page=2,
            subject_id="figure-1",
            modalities=["figure", "text"],
            status="partial",
            issues=["No matching value was found."],
        )
    ]
    state.quality_report.conflicts = [
        ConflictIssue(
            conflict_id="conflict-1",
            metric_name="results.score",
            values=["0.8", "0.9"],
            record_ids=["record-1", "record-2"],
            sources=["paper.pdf", "paper-b.pdf"],
            message="Sources report different values.",
        )
    ]
    state.coverage_report.gaps = [
        CoverageGap(
            gap_id="gap-1",
            requirement_name="experimental setup",
            priority="high",
            status="missing",
            reason="No evidence found.",
        )
    ]

    queue = build_review_queue(state)

    assert {item.subject_type for item in queue} == {
        "record",
        "figure",
        "cross_modal",
        "conflict",
        "coverage_gap",
    }
    assert len({item.review_id for item in queue}) == 5
    record_item = next(item for item in queue if item.subject_type == "record")
    assert record_item.evidence_refs == ["evidence-1"]
    assert record.raw == {"original_value": 0.8}
    assert queue[0].priority == "high"
