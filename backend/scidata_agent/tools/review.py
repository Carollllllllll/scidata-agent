from __future__ import annotations

import hashlib
from typing import Any

from scidata_agent.agent.schemas import AgentState, ReviewQueueItem


def build_review_queue(state: AgentState) -> list[ReviewQueueItem]:
    """Build one deterministic queue from all review-worthy audit signals.

    The queue is an index over existing evidence, not a second extraction
    result. It carries references and compact details while leaving every
    original record, chart value, conflict value, and coverage report intact.
    """

    items: list[ReviewQueueItem] = []
    seen: set[str] = set()
    evidence_by_record = {trace.record_id: trace.evidence_id for trace in state.evidence_traces}

    def add(item: ReviewQueueItem) -> None:
        if item.review_id not in seen:
            seen.add(item.review_id)
            items.append(item)

    for record in state.needs_review_records:
        reasons = list(record.warnings)
        raw_reasons = record.raw.get("needs_review_reasons") if isinstance(record.raw, dict) else None
        if isinstance(raw_reasons, list):
            reasons.extend(str(reason) for reason in raw_reasons if reason)
        add(ReviewQueueItem(
            review_id=_review_id("record", record.record_id, "record_quality"),
            subject_type="record",
            subject_id=record.record_id,
            record_id=record.record_id,
            priority="high" if record.confidence < 0.5 else "medium",
            risk_type="record_quality",
            title=f"Review extracted record {record.record_id}",
            reason=_join_reasons(reasons, "Record was flagged by quality or provenance checks."),
            source_file=record.source_file,
            page=record.page,
            evidence_refs=[evidence_by_record[record.record_id]] if record.record_id in evidence_by_record else [],
            details={"table_name": record.table_name, "confidence": record.confidence},
        ))

    for validation in state.chart_validations:
        if not validation.needs_review and not any(issue.severity in {"warning", "error"} for issue in validation.issues):
            continue
        add(ReviewQueueItem(
            review_id=_review_id("figure", validation.figure_id, "chart_validation"),
            subject_type="figure",
            subject_id=validation.figure_id,
            figure_id=validation.figure_id,
            priority="high" if any(issue.severity == "error" for issue in validation.issues) else "medium",
            risk_type="chart_validation",
            title=f"Review chart extraction {validation.figure_id}",
            reason=_join_reasons([issue.message for issue in validation.issues], "Chart validation flagged this figure."),
            details={"passed": validation.passed, "issues": [issue.model_dump(mode="json") for issue in validation.issues]},
        ))

    for check in state.cross_modal_checks:
        if check.status != "partial":
            continue
        add(ReviewQueueItem(
            review_id=_review_id("cross_modal", check.check_id, "cross_modal_partial"),
            subject_type="cross_modal",
            subject_id=check.check_id,
            priority="medium",
            risk_type="cross_modal_partial",
            title=f"Review cross-modal check {check.subject_id}",
            reason=_join_reasons(check.issues, "Text, table, and figure evidence only partially agree."),
            source_file=check.source_file,
            page=check.page,
            figure_id=check.subject_id if "figure" in check.modalities else None,
            evidence_refs=list(check.evidence_refs),
            details={
                "modalities": check.modalities,
                "matched_value_count": check.matched_value_count,
                "candidate_value_count": check.candidate_value_count,
                "confidence": check.confidence,
            },
        ))

    for conflict in state.quality_report.conflicts:
        add(ReviewQueueItem(
            review_id=_review_id("conflict", conflict.conflict_id, "source_conflict"),
            subject_type="conflict",
            subject_id=conflict.conflict_id,
            priority="high",
            risk_type="source_conflict",
            title=f"Review conflicting values for {conflict.metric_name}",
            reason=conflict.message,
            record_id=conflict.record_ids[0] if len(conflict.record_ids) == 1 else None,
            details={
                "metric_name": conflict.metric_name,
                "values": conflict.values,
                "record_ids": conflict.record_ids,
                "sources": conflict.sources,
                "alignment_context": conflict.alignment_context,
                "comparison_basis": conflict.comparison_basis,
                "resolution": conflict.resolution,
            },
        ))

    for gap in state.coverage_report.gaps:
        add(ReviewQueueItem(
            review_id=_review_id("coverage_gap", gap.gap_id, "coverage_gap"),
            subject_type="coverage_gap",
            subject_id=gap.gap_id,
            priority=gap.priority,
            risk_type="coverage_gap",
            title=f"Review evidence gap: {gap.requirement_name}",
            reason=gap.reason,
            details={
                "status": gap.status,
                "missing_fields": gap.missing_fields,
                "missing_evidence_types": gap.missing_evidence_types,
                "evidence_count": gap.evidence_count,
                "recommended_actions": gap.recommended_actions,
            },
        ))

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (priority_rank[item.priority], item.subject_type, item.subject_id))
    state.review_queue = items
    return items


def _review_id(subject_type: str, subject_id: str, risk_type: str) -> str:
    raw = f"{subject_type}|{subject_id}|{risk_type}"
    return f"review_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _join_reasons(reasons: list[Any], fallback: str) -> str:
    cleaned = [str(reason).strip() for reason in reasons if str(reason).strip()]
    return " | ".join(dict.fromkeys(cleaned)) or fallback
