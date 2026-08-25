from __future__ import annotations

from scidata_agent.agent.schemas import DynamicRecord
from scidata_agent.tools.quality import build_quality_report, detect_dynamic_conflicts


def _record(record_id: str, source_file: str, *, dataset: str = "benchmark", score: float = 0.8, method: str = "model-a") -> DynamicRecord:
    return DynamicRecord(
        record_id=record_id,
        table_name="results",
        fields={"method": method, "dataset": dataset, "score": score},
        source_file=source_file,
        evidence_text=f"{method} on {dataset}: score={score}",
    )


def test_dynamic_conflict_requires_aligned_context_and_preserves_values() -> None:
    records = [
        _record("r1", "paper-a.pdf", score=0.80),
        _record("r2", "paper-b.pdf", score=0.84),
    ]

    conflicts = detect_dynamic_conflicts(records, mutate_records=True)

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.metric_name == "results.score"
    assert conflict.values == ["0.8", "0.84"]
    assert conflict.alignment_context["dataset"] == "benchmark"
    assert conflict.resolution == "preserve_all"
    assert conflict.comparison_basis == ["table_name", "dataset", "method", "score"]
    assert records[0].raw["conflict_group_ids"] == [conflict.conflict_id]
    assert records[1].raw["conflict_group_ids"] == [conflict.conflict_id]


def test_dynamic_conflict_does_not_merge_different_conditions() -> None:
    records = [
        _record("r1", "paper-a.pdf", dataset="benchmark-a", score=0.80),
        _record("r2", "paper-b.pdf", dataset="benchmark-b", score=0.84),
    ]

    assert detect_dynamic_conflicts(records) == []


def test_dynamic_conflict_does_not_claim_comparability_without_context() -> None:
    records = [
        DynamicRecord(
            record_id="r1",
            table_name="results",
            fields={"method": "model-a", "score": 0.80},
            source_file="paper-a.pdf",
        ),
        DynamicRecord(
            record_id="r2",
            table_name="results",
            fields={"method": "model-a", "score": 0.84},
            source_file="paper-b.pdf",
        ),
    ]

    assert detect_dynamic_conflicts(records) == []


def test_quality_report_includes_dynamic_conflicts() -> None:
    records = [
        _record("r1", "paper-a.pdf", score=0.80),
        _record("r2", "paper-b.pdf", score=0.84),
    ]

    report = build_quality_report([], dynamic_records=records, mutate_records=True)

    assert report.conflict_count == 1
    assert report.conflicts[0].metric_name == "results.score"
    assert records[0].raw["conflict_group_ids"]
