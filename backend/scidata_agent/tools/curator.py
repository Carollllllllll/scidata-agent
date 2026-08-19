from __future__ import annotations

import copy
import re
from typing import Any

from scidata_agent.agent.schemas import DynamicRecord, SourceDiscoveryPlan


def curate_dynamic_records(
    records: list[DynamicRecord],
    source_discovery_plan: SourceDiscoveryPlan | None = None,
) -> tuple[list[DynamicRecord], list[DynamicRecord]]:
    """Return cleaned dynamic records plus records that need human review.

    The raw extraction output is intentionally preserved elsewhere. This layer
    performs deterministic result governance: arXiv metadata normalization,
    duplicate merging, and obvious weak-value isolation.
    """
    metadata_records = _metadata_records_from_arxiv(source_discovery_plan)
    repaired_records: list[DynamicRecord] = []
    review_records: list[DynamicRecord] = []

    for record in records:
        repaired = copy.deepcopy(record)
        review_reasons = _repair_dynamic_record(repaired)
        if review_reasons:
            repaired.raw["needs_review_reasons"] = review_reasons
            repaired.warnings.extend(reason for reason in review_reasons if reason not in repaired.warnings)
            review_records.append(copy.deepcopy(repaired))
        repaired_records.append(repaired)

    clean_records = _merge_dynamic_records(metadata_records + repaired_records)
    return clean_records, _merge_dynamic_records(review_records)


def _metadata_records_from_arxiv(source_discovery_plan: SourceDiscoveryPlan | None) -> list[DynamicRecord]:
    if not source_discovery_plan:
        return []
    records: list[DynamicRecord] = []
    for source in source_discovery_plan.candidate_sources:
        if source.source_type != "paper" or source.metadata.get("provider") != "arxiv":
            continue
        downloaded_path = source.metadata.get("downloaded_path")
        if not downloaded_path:
            continue
        source_file = str(downloaded_path).replace("\\", "/").split("/")[-1]
        fields = {
            "title": source.title,
            "authors": source.metadata.get("authors") or [],
            "publication_date": _date_only(source.metadata.get("published")),
            "venue": _arxiv_venue(source.url),
        }
        records.append(
            DynamicRecord(
                table_name="paper_metadata",
                fields=fields,
                source_file=source_file,
                source_type="pdf_text",
                page=None,
                evidence_text=f"arXiv metadata: {source.url or source.title}",
                confidence=1.0,
                warnings=[],
                raw={
                    "curation_source": "arxiv_metadata",
                    "arxiv_url": source.url,
                    "pdf_url": source.metadata.get("pdf_url"),
                    "published": source.metadata.get("published"),
                    "updated": source.metadata.get("updated"),
                },
            )
        )
    return records


def _repair_dynamic_record(record: DynamicRecord) -> list[str]:
    reasons: list[str] = []
    warning_text = " ".join(str(warning).lower() for warning in record.warnings)

    if record.table_name == "paper_metadata" and record.raw.get("curation_source") != "arxiv_metadata":
        if _weak_metadata_record(record):
            reasons.append("weak paper metadata from page-level LLM extraction; arXiv metadata should be preferred")

    if _warning_says_value_should_be_null(warning_text):
        changed_fields = []
        for field_name, value in list(record.fields.items()):
            if value not in (None, "", []):
                record.raw.setdefault("repaired_values", {})[field_name] = value
                record.fields[field_name] = None
                changed_fields.append(field_name)
        if changed_fields:
            reasons.append(f"fields nulled because warning says values are unreliable: {', '.join(changed_fields)}")

    if record.table_name == "deployment_efficiency":
        reasons.extend(_repair_deployment_efficiency(record))

    if not record.evidence_text and any(value not in (None, "", []) for value in record.fields.values()):
        reasons.append("non-empty dynamic record has no evidence_text")
        record.confidence = min(record.confidence, 0.55)

    return list(dict.fromkeys(reasons))


def _repair_deployment_efficiency(record: DynamicRecord) -> list[str]:
    reasons: list[str] = []
    evidence = record.evidence_text or ""
    warning_text = " ".join(str(warning).lower() for warning in record.warnings)

    latency = record.fields.get("inference_latency_ms")
    if isinstance(latency, int | float) and "8 seconds" in evidence.lower() and latency == 8000:
        record.raw.setdefault("repaired_values", {})["inference_latency_ms"] = latency
        record.fields["inference_latency_ms"] = 8000.0
        record.fields.setdefault("latency_unit_normalized", "ms")

    for field_name in ["inference_latency_ms", "model_size_mb", "memory_footprint_mb", "fps"]:
        value = record.fields.get(field_name)
        if value in (None, "", []):
            continue
        if _warning_says_value_should_be_null(warning_text):
            record.raw.setdefault("repaired_values", {})[field_name] = value
            record.fields[field_name] = None
            reasons.append(f"{field_name} moved to raw because extraction warning marked it unreliable")

    if record.fields.get("model_size_mb") and "parameters" in evidence.lower() and "mb" not in evidence.lower():
        record.raw.setdefault("repaired_values", {})["model_size_mb"] = record.fields["model_size_mb"]
        record.fields["model_size_mb"] = None
        reasons.append("model_size_mb inferred from parameter count rather than explicitly reported")

    if record.fields.get("memory_footprint_mb") and re.search(r"\bgb\b", evidence, flags=re.IGNORECASE):
        reasons.append("memory footprint evidence is in GB; unit conversion should be reviewed")

    return reasons


def _merge_dynamic_records(records: list[DynamicRecord]) -> list[DynamicRecord]:
    merged: dict[tuple[Any, ...], DynamicRecord] = {}
    for record in records:
        key = _dynamic_record_key(record)
        if key not in merged:
            merged[key] = copy.deepcopy(record)
            merged[key].raw.setdefault("merged_record_ids", [record.record_id])
            continue
        target = merged[key]
        target.raw.setdefault("merged_record_ids", [])
        if record.record_id not in target.raw["merged_record_ids"]:
            target.raw["merged_record_ids"].append(record.record_id)
        _merge_into(target, record)
    return list(merged.values())


def _merge_into(target: DynamicRecord, source: DynamicRecord) -> None:
    for field_name, value in source.fields.items():
        current = target.fields.get(field_name)
        if current in (None, "", []) and value not in (None, "", []):
            target.fields[field_name] = value
    if source.confidence > target.confidence:
        target.confidence = source.confidence
    if source.evidence_text and source.evidence_text != target.evidence_text:
        evidence_list = target.raw.setdefault("evidence_list", [])
        if target.evidence_text and target.evidence_text not in evidence_list:
            evidence_list.append(target.evidence_text)
        if source.evidence_text not in evidence_list:
            evidence_list.append(source.evidence_text)
    for warning in source.warnings:
        if warning not in target.warnings:
            target.warnings.append(warning)
    for key, value in source.raw.items():
        if key not in target.raw:
            target.raw[key] = value


def _dynamic_record_key(record: DynamicRecord) -> tuple[Any, ...]:
    fields = record.fields
    source_file = _norm(record.source_file)
    table = record.table_name
    if table == "paper_metadata":
        return (table, source_file)
    if table == "method_architecture":
        return (table, source_file, _norm(fields.get("architecture_type")), _norm_list(fields.get("backbone_names")))
    if table in {"key_module", "key_modules"}:
        return (table, source_file, _norm(fields.get("module_name")))
    if table == "dataset_usage":
        return (table, source_file, _norm(fields.get("dataset_name")), _norm(fields.get("usage_purpose")))
    if table in {"evaluation_metrics", "metric_result"}:
        return (
            table,
            source_file,
            _norm(fields.get("metric_name")),
            _norm(fields.get("dataset_split")),
            _norm(fields.get("method_name")),
            _norm(fields.get("baseline_comparison")),
        )
    if table == "deployment_efficiency":
        populated = tuple(sorted(key for key, value in fields.items() if value not in (None, "", [])))
        return (table, source_file, populated, record.page)
    return (table, source_file, tuple(sorted((key, _norm(value)) for key, value in fields.items())))


def _weak_metadata_record(record: DynamicRecord) -> bool:
    fields = record.fields
    authors = fields.get("authors")
    publication_date = fields.get("publication_date")
    if not publication_date:
        return True
    if authors in (None, "", [], ["unknown"], ["Unknown"], "unknown", "Unknown"):
        return True
    if any("unknown" in str(author).lower() for author in authors) if isinstance(authors, list) else "unknown" in str(authors).lower():
        return True
    return False


def _warning_says_value_should_be_null(warning_text: str) -> bool:
    phrases = [
        "should be null",
        "must be null",
        "cannot be reliably extracted",
        "not reliably extracted",
        "placeholder is invalid",
        "per missing_data_policy",
        "not explicitly given",
    ]
    return any(phrase in warning_text for phrase in phrases)


def _date_only(value: Any) -> str | None:
    if not value:
        return None
    return str(value)[:10]


def _arxiv_venue(url: str | None) -> str:
    if not url:
        return "arXiv"
    arxiv_id = str(url).rstrip("/").split("/")[-1]
    return f"arXiv:{arxiv_id}" if arxiv_id else "arXiv"


def _norm(value: Any) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, list):
        return ";".join(_norm(item) for item in value)
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _norm_list(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(sorted(_norm(item) for item in value if item not in (None, "")))
    return _norm(value)
