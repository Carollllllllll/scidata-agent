from __future__ import annotations

import re
from collections import defaultdict

from scidata_agent.agent.schemas import ConflictIssue, QualityIssue, QualityReport, ScientificRecord


DIMENSIONLESS_METRICS = {
    "accuracy",
    "acc",
    "precision",
    "recall",
    "f1",
    "f1-score",
    "auc",
    "auroc",
    "ssim",
    "lpips",
    "fid",
    "kid",
    "clip score",
    "r2",
    "pearson",
    "spearman",
    "number of parameters",
    "parameters",
}

REQUIRED_FIELDS = [
    "metric_name",
    "source_file",
    "evidence_text",
    "confidence",
]


def build_quality_report(
    records: list[ScientificRecord],
    llm_issues: list[QualityIssue] | None = None,
    target_fields: list[str] | None = None,
) -> QualityReport:
    """Build the competition-facing quality report for the Data Agent loop.

    The report is intentionally stricter than a normal parser QA pass: values
    should be traceable to evidence, conflicts should be visible, and field
    coverage should be measurable.
    """

    issues: list[QualityIssue] = []
    for record in records:
        issues.extend(_check_record(record))

    if llm_issues:
        issues.extend(llm_issues)

    conflicts = detect_conflicts(records)
    for conflict in conflicts:
        issues.append(
            QualityIssue(
                level="warning",
                field="metric_value",
                message=conflict.message,
            )
        )

    evidence_count = sum(1 for record in records if record.evidence_text)
    value_supported_count = sum(1 for record in records if _value_supported_by_evidence(record))
    field_coverage = _field_coverage(records, target_fields or REQUIRED_FIELDS)
    source_count = len({record.source_file for record in records if record.source_file})
    warning_count = sum(1 for issue in issues if issue.level == "warning")
    error_count = sum(1 for issue in issues if issue.level == "error")

    notes = [
        "Quality report follows the Data Agent loop: provenance, schema coverage, evidence support, and conflict checks.",
        "A parser-only result is not sufficient for official evaluation; every critical value should keep source evidence.",
    ]
    if not records:
        notes.append("No records were extracted. Check whether the question matches the uploaded sources.")
    if conflicts:
        notes.append("Conflicts are retained instead of overwritten so reviewers can inspect source disagreement.")

    return QualityReport(
        record_count=len(records),
        issue_count=len(issues),
        warning_count=warning_count,
        error_count=error_count,
        conflict_count=len(conflicts),
        evidence_coverage=_ratio(evidence_count, len(records)),
        value_evidence_coverage=_ratio(value_supported_count, len(records)),
        field_coverage=field_coverage,
        source_count=source_count,
        issues=issues,
        conflicts=conflicts,
        notes=notes,
    )


def detect_conflicts(records: list[ScientificRecord]) -> list[ConflictIssue]:
    groups: dict[tuple[str, str, str], list[ScientificRecord]] = defaultdict(list)
    for record in records:
        entity = _entity_key(record)
        metric = _norm(record.metric_name)
        context = _context_key(record)
        if not metric or record.metric_value is None:
            continue
        groups[(entity, metric, context)].append(record)

    conflicts: list[ConflictIssue] = []
    for (entity, metric, context), group in groups.items():
        value_groups: dict[str, list[ScientificRecord]] = defaultdict(list)
        for record in group:
            value_groups[_value_key(record)].append(record)
        if len(value_groups) <= 1:
            continue
        representative_values = list(value_groups.keys())
        conflict_records = [record for records_for_value in value_groups.values() for record in records_for_value]
        sources = sorted({record.source_file for record in conflict_records if record.source_file})
        conflicts.append(
            ConflictIssue(
                entity=entity or None,
                metric_name=metric,
                values=representative_values,
                record_ids=[record.record_id for record in conflict_records],
                sources=sources,
                message=(
                    f"Potential conflict for entity='{entity or 'unknown'}', metric='{metric}': "
                    f"{', '.join(representative_values)} under context='{context or 'unspecified'}'."
                ),
            )
        )
    return conflicts


def _check_record(record: ScientificRecord) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    metric = _norm(record.metric_name)

    if not record.source_file:
        issues.append(_issue(record, "error", "source_file", "Record is missing source_file."))
        _warn(record, "missing source_file")

    if not record.evidence_text:
        issues.append(_issue(record, "warning", "evidence_text", "Record is missing evidence_text."))
        _warn(record, "missing evidence_text")
        record.confidence = min(record.confidence, 0.45)

    if record.metric_value is None:
        issues.append(_issue(record, "warning", "metric_value", "metric_value is null or cannot be parsed as a number."))
        _warn(record, "invalid metric_value")
        record.confidence = min(record.confidence, 0.55)
    elif record.evidence_text and not _value_supported_by_evidence(record):
        issues.append(
            _issue(
                record,
                "warning",
                "evidence_text",
                "metric_value is not directly visible in evidence_text; possible hallucination or weak provenance.",
            )
        )
        _warn(record, "metric_value not found in evidence_text")
        record.confidence = min(record.confidence, 0.5)

    if not record.unit:
        if metric in DIMENSIONLESS_METRICS:
            record.unit = "dimensionless"
        else:
            issues.append(_issue(record, "info", "unit", "unit is missing; metric may be unitless or source omitted it."))
            _warn(record, "unit missing")

    if record.confidence < 0.6:
        issues.append(_issue(record, "warning", "confidence", "Record confidence is low and should be reviewed."))

    return issues


def _value_supported_by_evidence(record: ScientificRecord) -> bool:
    if record.metric_value is None:
        return False
    if not record.evidence_text:
        return False
    evidence = _compact_number_text(record.evidence_text)
    candidates = _number_strings(record.metric_value)
    return any(candidate in evidence for candidate in candidates)


def _number_strings(value: float) -> set[str]:
    raw = float(value)
    candidates = {
        _compact_number_text(f"{raw}"),
        _compact_number_text(f"{raw:g}"),
        _compact_number_text(f"{raw:.1f}"),
        _compact_number_text(f"{raw:.2f}"),
        _compact_number_text(str(int(raw))) if raw.is_integer() else "",
    }
    return {candidate for candidate in candidates if candidate}


def _compact_number_text(text: str) -> str:
    return re.sub(r"[\s,%]+", "", text.lower())


def _field_coverage(records: list[ScientificRecord], fields: list[str]) -> dict[str, float]:
    coverage: dict[str, float] = {}
    unique_fields = [field for field in dict.fromkeys(fields) if hasattr(ScientificRecord, field) or field in ScientificRecord.model_fields]
    for field in unique_fields:
        present = sum(1 for record in records if getattr(record, field, None) not in (None, "", []))
        coverage[field] = _ratio(present, len(records))
    return coverage


def _issue(record: ScientificRecord, level: str, field: str, message: str) -> QualityIssue:
    return QualityIssue(record_id=record.record_id, level=level, field=field, message=message)  # type: ignore[arg-type]


def _warn(record: ScientificRecord, message: str) -> None:
    if message not in record.warnings:
        record.warnings.append(message)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _entity_key(record: ScientificRecord) -> str:
    return _norm(record.material or record.paper_title or record.method or "unknown")


def _context_key(record: ScientificRecord) -> str:
    raw = record.raw or {}
    attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
    parts = [
        record.method,
        record.condition,
        attributes.get("setting"),
        attributes.get("variant"),
        attributes.get("baseline"),
        attributes.get("dataset"),
        attributes.get("split"),
        attributes.get("task"),
    ]
    return _norm(" | ".join(str(part) for part in parts if part not in (None, "", [])))


def _value_key(record: ScientificRecord) -> str:
    value = f"{record.metric_value:g}" if record.metric_value is not None else "null"
    unit = record.unit or ""
    return f"{value} {unit}".strip()


def _norm(value: str) -> str:
    return " ".join(str(value).strip().lower().split())
