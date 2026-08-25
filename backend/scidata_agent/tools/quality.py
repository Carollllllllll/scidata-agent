from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scidata_agent.agent.schemas import (
    ConflictIssue,
    DynamicExtractionPlan,
    DynamicRecord,
    QualityIssue,
    QualityReport,
    ScientificRecord,
    TableBlock,
    TextBlock,
)


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
    dynamic_records: list[DynamicRecord] | None = None,
    dynamic_plan: DynamicExtractionPlan | None = None,
    text_blocks: list[TextBlock] | None = None,
    table_blocks: list[TableBlock] | None = None,
    *,
    mutate_records: bool = False,
) -> QualityReport:
    """Build the competition-facing quality report for the Data Agent loop.

    The report is intentionally stricter than a normal parser QA pass: values
    should be traceable to evidence, conflicts should be visible, and field
    coverage should be measurable.
    """

    checked_records = records if mutate_records else [record.model_copy(deep=True) for record in records]
    supplied_dynamic_records = dynamic_records or []
    checked_dynamic_records = (
        supplied_dynamic_records
        if mutate_records
        else [record.model_copy(deep=True) for record in supplied_dynamic_records]
    )

    issues: list[QualityIssue] = []
    for record in checked_records:
        issues.extend(_check_record(record))

    if llm_issues:
        issues.extend(llm_issues)

    required_fields = _dynamic_required_fields(dynamic_plan)
    for record in checked_dynamic_records:
        issues.extend(_check_dynamic_record(record, required_fields.get(record.table_name, set())))

    provenance_matches, provenance_total, provenance_issues = _validate_provenance_pages(
        [*checked_records, *checked_dynamic_records], text_blocks or [], table_blocks or []
    )
    issues.extend(provenance_issues)

    conflicts = detect_conflicts(checked_records)
    conflicts.extend(detect_dynamic_conflicts(checked_dynamic_records, mutate_records=mutate_records))
    for conflict in conflicts:
        issues.append(
            QualityIssue(
                level="warning",
                field="metric_value",
                message=conflict.message,
            )
        )

    evidence_count = sum(1 for record in checked_records if record.evidence_text)
    all_records: list[Any] = [*checked_records, *checked_dynamic_records]
    all_evidence_count = sum(1 for record in all_records if record.evidence_text)
    value_supported_count = sum(1 for record in checked_records if _value_supported_by_evidence(record))
    field_coverage = _field_coverage(checked_records, target_fields or REQUIRED_FIELDS)
    source_count = len({record.source_file for record in checked_records if record.source_file})
    warning_count = sum(1 for issue in issues if issue.level == "warning")
    error_count = sum(1 for issue in issues if issue.level == "error")
    review_ids = {issue.record_id for issue in issues if issue.record_id and issue.level in {"warning", "error"}}
    warning_free_count = sum(1 for record in all_records if record.record_id not in review_ids)

    notes = [
        "Quality report follows the Data Agent loop: provenance, schema coverage, evidence support, and conflict checks.",
        "A parser-only result is not sufficient for official evaluation; every critical value should keep source evidence.",
    ]
    if not checked_records:
        notes.append("No records were extracted. Check whether the question matches the uploaded sources.")
    if conflicts:
        notes.append("Conflicts are retained instead of overwritten so reviewers can inspect source disagreement.")

    return QualityReport(
        record_count=len(checked_records),
        dynamic_record_count=len(checked_dynamic_records),
        total_record_count=len(all_records),
        issue_count=len(issues),
        warning_count=warning_count,
        error_count=error_count,
        conflict_count=len(conflicts),
        evidence_coverage=_ratio(evidence_count, len(checked_records)),
        evidence_text_coverage=_ratio(all_evidence_count, len(all_records)),
        value_evidence_coverage=_ratio(value_supported_count, len(checked_records)),
        provenance_page_coverage=_ratio(provenance_matches, provenance_total),
        warning_free_rate=_ratio(warning_free_count, len(all_records)),
        review_count=len(review_ids),
        field_coverage=field_coverage,
        source_count=source_count,
        issues=issues,
        conflicts=conflicts,
        notes=notes,
    )


def _dynamic_required_fields(dynamic_plan: DynamicExtractionPlan | None) -> dict[str, set[str]]:
    if dynamic_plan is None:
        return {}
    return {
        table.table_name: {field.name for field in table.fields if field.required}
        for table in dynamic_plan.dynamic_tables
    }


def _check_dynamic_record(record: DynamicRecord, required_fields: set[str]) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    seen_messages: set[tuple[str, str]] = set()

    def add(level: str, field: str, message: str) -> None:
        key = (field.casefold(), " ".join(message.casefold().rstrip(".").split()))
        if key in seen_messages:
            return
        seen_messages.add(key)
        issues.append(
            QualityIssue(record_id=record.record_id, level=level, field=field, message=message)  # type: ignore[arg-type]
        )

    for field in sorted(required_fields):
        if record.fields.get(field) in (None, "", []):
            add("warning", field, f"Required dynamic field is missing: {field}.")
    if not record.evidence_text and any(value not in (None, "", []) for value in record.fields.values()):
        add("warning", "evidence_text", "Dynamic record has values but no evidence_text.")
    if record.confidence < 0.6:
        add("warning", "confidence", "Dynamic record confidence is low and should be reviewed.")
    for warning in record.warnings:
        normalized_warning = " ".join(str(warning).casefold().rstrip(".").split())
        missing_match = re.fullmatch(r"required dynamic field missing:\s*([a-z0-9_]+)", normalized_warning)
        if missing_match and missing_match.group(1) in required_fields:
            # The schema-derived issue above is the canonical representation.
            continue
        if normalized_warning == "record contains extraction warnings that require review":
            # This is a curator routing marker, not an independent defect.
            continue
        add("warning", "record_warning", str(warning))
    return issues


def _validate_provenance_pages(
    records: list[ScientificRecord | DynamicRecord],
    text_blocks: list[TextBlock],
    table_blocks: list[TableBlock] | None = None,
) -> tuple[int, int, list[QualityIssue]]:
    page_index: dict[tuple[str, int], str] = {}
    for block in text_blocks:
        if block.page is None:
            continue
        for source_name in {block.source_file, Path(block.source_file).name}:
            key = (source_name.casefold(), block.page)
            page_index[key] = f"{page_index.get(key, '')}\n{block.text}"
    for table in table_blocks or []:
        if table.page is None:
            continue
        table_lines = [table.caption or "", " ".join(table.columns)]
        for row in table.rows:
            table_lines.append(
                ", ".join(
                    f"{column}: {json.dumps(row.get(column), ensure_ascii=False)}"
                    for column in table.columns
                )
            )
        table_text = "\n".join(line for line in table_lines if line)
        for source_name in {table.source_file, Path(table.source_file).name}:
            key = (source_name.casefold(), table.page)
            page_index[key] = f"{page_index.get(key, '')}\n{table_text}"

    matches = 0
    total = 0
    issues: list[QualityIssue] = []
    for record in records:
        source_type = str(getattr(record.source_type, "value", record.source_type)).lower()
        if source_type not in {"pdf_text", "pdf_table"} or not record.evidence_text:
            continue
        total += 1
        page = record.page
        page_text = None
        if page is not None:
            source_name = Path(record.source_file).name.casefold()
            page_text = page_index.get((record.source_file.casefold(), page)) or page_index.get((source_name, page))
        evidence = _compact_provenance_text(record.evidence_text)
        page_content = _compact_provenance_text(page_text or "")
        if page is not None and len(evidence) >= 8 and _evidence_matches_page(record.evidence_text, page_text or ""):
            matches += 1
            continue

        message = (
            "Evidence text was not found on the recorded PDF page; verify page provenance."
            if page is not None
            else "PDF evidence has no recorded page; verify page provenance."
        )
        issues.append(
            QualityIssue(
                record_id=record.record_id,
                level="warning",
                field="page",
                message=message,
            )
        )
        warning = "evidence text not found on recorded PDF page"
        if warning not in record.warnings:
            record.warnings.append(warning)
        record.confidence = min(record.confidence, 0.5)
    return matches, total, issues


def _compact_provenance_text(text: str) -> str:
    return "".join(character.casefold() for character in str(text) if character.isalnum())


def _evidence_matches_page(evidence_text: str, page_text: str) -> bool:
    evidence = _compact_provenance_text(evidence_text)
    page = _compact_provenance_text(page_text)
    if evidence and evidence in page:
        return True

    # Table extractors often cite a header plus one row while other rows occur
    # between them in page reading order. Accept that non-contiguous form only
    # when every meaningful token appears in order on the recorded page.
    evidence_tokens = _provenance_tokens(evidence_text)
    page_tokens = _provenance_tokens(page_text)
    if len(evidence_tokens) < 4:
        return False
    has_numeric_token = any(token[0].isdigit() for token in evidence_tokens if token)
    if not has_numeric_token and len(evidence_tokens) < 8:
        return False
    cursor = 0
    for token in evidence_tokens:
        try:
            cursor = page_tokens.index(token, cursor) + 1
        except ValueError:
            return False
    return True


def _provenance_tokens(text: str) -> list[str]:
    return re.findall(
        r"\d+(?:\.\d+)*|[a-z]+[a-z0-9]*(?:-[a-z0-9]+)*|[\u3400-\u9fff]+",
        str(text).casefold(),
    )


def detect_conflicts(records: list[ScientificRecord]) -> list[ConflictIssue]:
    groups: dict[tuple[str, str, str], list[ScientificRecord]] = defaultdict(list)
    for record in records:
        entity = _entity_key(record)
        metric = _norm(record.metric_name)
        context = _context_key(record)
        if not metric or record.metric_value is None:
            continue
        if not context:
            # Several values without dataset/method/condition are ambiguous,
            # but there is not enough information to assert a contradiction.
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
                alignment_context={"context": context},
                comparison_basis=["entity", "metric_name", "metric_value", "unit", "condition"],
            )
        )
    return conflicts


_DYNAMIC_CONTEXT_KEYS = {
    "condition", "setting", "scenario", "dataset", "split", "task",
    "environment", "protocol", "variant", "baseline", "regime", "configuration",
}
_DYNAMIC_ENTITY_KEYS = {
    "entity", "material", "model", "method", "name", "id", "system", "group", "category",
}
_DYNAMIC_IGNORED_KEYS = {
    "title", "author", "authors", "date", "year", "page", "source", "citation", "evidence", "confidence",
}
_DYNAMIC_VALUE_HINTS = {
    "metric", "score", "value", "result", "accuracy", "precision", "recall", "loss", "error", "rate",
    "latency", "time", "memory", "size", "count", "number",
}


def detect_dynamic_conflicts(
    records: list[DynamicRecord],
    *,
    mutate_records: bool = False,
) -> list[ConflictIssue]:
    """Compare dynamic records only when their generated conditions align.

    The field names remain task-specific and come from the LLM extraction plan.
    The alias sets describe structural roles (condition/entity/value), not a
    scientific domain schema. Missing alignment context is non-comparable.
    """
    groups: dict[tuple[str, tuple[tuple[str, str], ...], str], list[DynamicRecord]] = defaultdict(list)
    for record in records:
        context = _dynamic_alignment_context(record.fields)
        if not context:
            continue
        identity = tuple(sorted(context.items()))
        for field_name, value in record.fields.items():
            if not _dynamic_value_field(field_name, value):
                continue
            groups[(record.table_name, identity, _dynamic_field_key(field_name))].append(record)

    conflicts: list[ConflictIssue] = []
    for (table_name, identity, field_name), group in groups.items():
        sources = sorted({record.source_file for record in group if record.source_file})
        if len(sources) < 2:
            continue
        value_groups: dict[str, list[DynamicRecord]] = defaultdict(list)
        display_values: dict[str, str] = {}
        for record in group:
            value = next(
                (value for key, value in record.fields.items() if _dynamic_field_key(key) == field_name),
                None,
            )
            value_key = _dynamic_value_key(value)
            if not value_key:
                continue
            value_groups[value_key].append(record)
            display_values.setdefault(value_key, _display_dynamic_value(value))
        if len(value_groups) <= 1:
            continue

        conflict_records = [record for records_for_value in value_groups.values() for record in records_for_value]
        context = dict(identity)
        conflict_id = "dynamic_conflict_" + hashlib.sha1(
            repr((table_name, identity, field_name)).encode("utf-8")
        ).hexdigest()[:10]
        conflicts.append(
            ConflictIssue(
                conflict_id=conflict_id,
                entity=_dynamic_entity_label(context),
                metric_name=f"{table_name}.{field_name}",
                values=[display_values[key] for key in value_groups],
                record_ids=[record.record_id for record in conflict_records],
                sources=sources,
                message=(
                    f"Dynamic field '{field_name}' differs across sources for the aligned "
                    f"context {context}: {', '.join(display_values.values())}. "
                    "All values are preserved for review."
                ),
                alignment_context=context,
                comparison_basis=["table_name", *context.keys(), field_name],
                resolution="preserve_all",
            )
        )
        if mutate_records:
            for record in conflict_records:
                record.raw.setdefault("conflict_group_ids", []).append(conflict_id)
    return conflicts


def _dynamic_alignment_context(fields: dict[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {}
    for key, value in fields.items():
        tokens = _dynamic_field_tokens(key)
        if not _has_value(value) or not tokens:
            continue
        if tokens & _DYNAMIC_CONTEXT_KEYS or tokens & _DYNAMIC_ENTITY_KEYS:
            context[_dynamic_field_key(key)] = _display_dynamic_value(value)
    if not any(set(key.split("_")) & _DYNAMIC_CONTEXT_KEYS for key in context):
        return {}
    return context


def _dynamic_value_field(field_name: str, value: Any) -> bool:
    if not _has_value(value) or isinstance(value, (dict, list, tuple, set)):
        return False
    tokens = _dynamic_field_tokens(field_name)
    if tokens & _DYNAMIC_IGNORED_KEYS:
        return False
    return bool(tokens & _DYNAMIC_VALUE_HINTS) or isinstance(value, (int, float))


def _dynamic_field_tokens(field_name: str) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(field_name))
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _dynamic_field_key(field_name: str) -> str:
    return "_".join(sorted(_dynamic_field_tokens(field_name))) or str(field_name).strip().casefold()


def _dynamic_value_key(value: Any) -> str:
    if not _has_value(value):
        return ""
    if isinstance(value, bool):
        return str(value).casefold()
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _display_dynamic_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _dynamic_entity_label(context: dict[str, str]) -> str | None:
    entity_items = [
        f"{key}={value}"
        for key, value in context.items()
        if set(key.split("_")) & _DYNAMIC_ENTITY_KEYS
    ]
    return "; ".join(entity_items) or None


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {}, ())


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
    return any(
        re.search(rf"(?<![\d.]){re.escape(candidate)}(?!\d|\.\d)", evidence) is not None
        for candidate in candidates
    )


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
