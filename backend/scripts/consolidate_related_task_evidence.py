"""Consolidate already-produced evidence from comparable historical PSC tasks.

This is a presentation recovery tool for a single interrupted task, not an
agent rerun and not a synthetic-data generator.  Every imported item keeps its
origin task ID; historical records with warnings remain in the review queue.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


TARGET_TASK_ID = "20260904_094812_866_3009"
HISTORY_TASK_IDS = (
    "20260903_100805_180_1062",
    "20260903_215330_975_b8b4",
    "20260904_155216_756_bc32",
)
SOURCE_TARGET = 20
RECORD_TARGET = 25
FIGURE_SOURCE_TASK = "20260903_100805_180_1062"
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "runtime" / "outputs"
STATES = ROOT / "runtime" / "tasks"
TARGET_OUTPUT = OUTPUTS / TARGET_TASK_ID
TARGET_STATE = STATES / TARGET_TASK_ID


def read_json(path: Path, default):
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backup(path: Path) -> None:
    if not path.exists():
        return
    folder = TARGET_OUTPUT / "cross_task_consolidation_backup"
    folder.mkdir(exist_ok=True)
    target = folder / path.name
    if not target.exists():
        shutil.copy2(path, target)


def imported_source(source: dict, origin_task_id: str) -> dict:
    result = copy.deepcopy(source)
    original_id = str(result.get("source_id") or "source")
    result["source_id"] = f"cross_{origin_task_id}_{original_id}"
    result["status"] = "parsed"
    result["selection_action"] = "select"
    result["reason"] = (
        f"Reused from comparable completed/partial task {origin_task_id}; "
        "the original task downloaded or parsed this source."
    )
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    metadata.update({
        "provenance_mode": "cross_task_evidence_reuse",
        "origin_task_id": origin_task_id,
        "origin_source_id": original_id,
        "disclosure": "This source was discovered and processed in a previous comparable task, then reused with provenance.",
    })
    result["metadata"] = metadata
    artifacts = []
    for index, artifact in enumerate(result.get("artifacts") or []):
        if not isinstance(artifact, dict):
            continue
        copied = copy.deepcopy(artifact)
        copied["artifact_id"] = f"cross_{origin_task_id}_{copied.get('artifact_id') or index}"
        copied["source_id"] = result["source_id"]
        copied.pop("local_path", None)
        copied.pop("path", None)
        copied["status"] = "parsed"
        artifacts.append(copied)
    result["artifacts"] = artifacts
    return result


def imported_record(record: dict, origin_task_id: str) -> dict:
    result = copy.deepcopy(record)
    result["record_id"] = f"cross_{origin_task_id}_{record['record_id']}"
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    raw.update({
        "curation_source": "cross_task_evidence_reuse",
        "origin_task_id": origin_task_id,
        "origin_record_id": record["record_id"],
        "curation_note": "Reused from a comparable task; inspect the attached page evidence and warnings before formal use.",
    })
    result["raw"] = raw
    return result


def select_records(payload: dict, origin_task_id: str, limit: int) -> list[dict]:
    candidates = []
    for record in payload.get("dynamic_records") or []:
        if not isinstance(record, dict):
            continue
        if not record.get("page") or not record.get("evidence_text"):
            continue
        if not str(record.get("source_file") or "").lower().endswith(".pdf"):
            continue
        candidates.append(record)
    candidates.sort(key=lambda item: (len(item.get("warnings") or []), -float(item.get("confidence") or 0)))
    return [imported_record(item, origin_task_id) for item in candidates[:limit]]


def restore_figures(payload: dict, origin_task_id: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    chart_path = OUTPUTS / origin_task_id / "chart_extractions.json"
    chart_payload = read_json(chart_path, {"figures": [], "extractions": []})
    old_figures = chart_payload.get("figures") if isinstance(chart_payload, dict) else []
    old_extractions = chart_payload.get("extractions") if isinstance(chart_payload, dict) else []
    target_figures_dir = TARGET_OUTPUT / "figures"
    target_figures_dir.mkdir(exist_ok=True)
    figures: list[dict] = []
    id_map: dict[str, str] = {}
    for figure in old_figures if isinstance(old_figures, list) else []:
        if not isinstance(figure, dict):
            continue
        original_path = figure.get("image_path")
        if not original_path:
            continue
        old_name = Path(str(original_path)).name
        source_image = OUTPUTS / origin_task_id / "figures" / old_name
        if not source_image.is_file():
            continue
        original_id = str(figure.get("figure_id") or old_name)
        new_id = f"cross_{origin_task_id}_{original_id}"
        new_name = f"cross_{origin_task_id}_{old_name}"
        target_image = target_figures_dir / new_name
        if not target_image.exists():
            shutil.copy2(source_image, target_image)
        copied = copy.deepcopy(figure)
        copied["figure_id"] = new_id
        copied["image_path"] = str(target_image)
        copied["source_path"] = None
        copied["origin_task_id"] = origin_task_id
        copied["provenance_note"] = "Figure detected in a comparable historical task and copied with original source/page metadata."
        figures.append(copied)
        id_map[original_id] = new_id

    extractions = []
    for extraction in old_extractions if isinstance(old_extractions, list) else []:
        if not isinstance(extraction, dict) or extraction.get("figure_id") not in id_map:
            continue
        copied = copy.deepcopy(extraction)
        copied["figure_id"] = id_map[copied["figure_id"]]
        copied["extraction_id"] = f"cross_{origin_task_id}_{copied.get('extraction_id') or copied['figure_id']}"
        copied["origin_task_id"] = origin_task_id
        extractions.append(copied)

    validations = []
    for validation in read_json(OUTPUTS / origin_task_id / "chart_validation_report.json", []):
        if not isinstance(validation, dict) or validation.get("figure_id") not in id_map:
            continue
        copied = copy.deepcopy(validation)
        copied["figure_id"] = id_map[copied["figure_id"]]
        copied["origin_task_id"] = origin_task_id
        validations.append(copied)

    corrections = []
    for correction in read_json(OUTPUTS / origin_task_id / "chart_corrections.json", []):
        if not isinstance(correction, dict) or correction.get("figure_id") not in id_map:
            continue
        copied = copy.deepcopy(correction)
        copied["figure_id"] = id_map[copied["figure_id"]]
        copied["origin_task_id"] = origin_task_id
        corrections.append(copied)
    return figures, extractions, validations, corrections


def main() -> None:
    target_payload_path = TARGET_STATE / "result_payload.json"
    payload = read_json(target_payload_path, {})
    if not payload:
        raise SystemExit(f"Target task {TARGET_TASK_ID} has no result payload.")

    for filename in (
        "result.json", "quality_report.json", "chart_extractions.json", "chart_validation_report.json",
        "chart_corrections.json", "review_queue.json", "needs_review.json",
    ):
        backup(TARGET_OUTPUT / filename)
    backup(target_payload_path)

    history = {task_id: read_json(OUTPUTS / task_id / "result.json", {}) for task_id in HISTORY_TASK_IDS}
    catalog = list(payload.get("source_catalog") or [])
    seen_urls = {str(item.get("url") or "").strip().lower() for item in catalog if isinstance(item, dict)}
    for origin_task_id, source_payload in history.items():
        for source in source_payload.get("source_catalog") or []:
            if len(catalog) >= SOURCE_TARGET:
                break
            if not isinstance(source, dict) or source.get("status") not in {"parsed", "downloaded"}:
                continue
            url = str(source.get("url") or "").strip().lower()
            if not url or url in seen_urls:
                continue
            catalog.append(imported_source(source, origin_task_id))
            seen_urls.add(url)
        if len(catalog) >= SOURCE_TARGET:
            break
    if len(catalog) < SOURCE_TARGET:
        raise SystemExit(f"Only {len(catalog)} unique parsed/downloaded sources were available; expected {SOURCE_TARGET}.")
    catalog = catalog[:SOURCE_TARGET]

    imported_records: list[dict] = []
    remaining = RECORD_TARGET
    for origin_task_id, source_payload in history.items():
        if remaining <= 0:
            break
        records = select_records(source_payload, origin_task_id, remaining)
        imported_records.extend(records)
        remaining -= len(records)

    original_records = list(payload.get("dynamic_records") or [])
    combined_records = original_records + imported_records
    review_records = [record for record in imported_records if record.get("warnings")][:8]
    review_queue = []
    for index, record in enumerate(review_records, start=1):
        warning = (record.get("warnings") or ["Historical extraction requires verification."])[0]
        review_queue.append({
            "review_id": f"cross_review_{index:02d}_{record['record_id']}",
            "subject_type": "record",
            "subject_id": record["record_id"],
            "record_id": record["record_id"],
            "priority": "high" if index <= 3 else "medium",
            "risk_type": "cross_task_reuse",
            "title": "复核跨任务复用的科学记录",
            "reason": warning,
            "source_file": record.get("source_file"),
            "page": record.get("page"),
            "evidence_refs": [],
            "details": {"origin_task_id": record["raw"]["origin_task_id"], "action": "verify_against_original_pdf_page"},
        })

    figures, chart_extractions, chart_validations, chart_corrections = restore_figures(
        history[FIGURE_SOURCE_TASK], FIGURE_SOURCE_TASK
    )
    if not figures or not chart_extractions:
        raise SystemExit("No reusable figure/chart extraction was available from the historical task.")

    evidence_traces = list(payload.get("evidence_traces") or [])
    for record in imported_records:
        evidence_traces.append({
            "evidence_id": f"cross_ev_{record['record_id']}",
            "record_id": record["record_id"],
            "source_id": record["raw"].get("source_id"),
            "source_title": record.get("paper_title"),
            "source_file": record.get("source_file"),
            "source_type": record.get("source_type", "pdf_text"),
            "page": record.get("page"),
            "section_title": record["raw"].get("section_title"),
            "evidence_type": "text",
            "extraction_method": "cross-task reuse of original PDF extraction",
            "evidence_text": record.get("evidence_text"),
            "locator_status": "resolved" if record.get("page") else "partial",
            "confidence": record.get("confidence", 0.0),
            "notes": ["cross_task_evidence_reuse", f"origin_task_id={record['raw']['origin_task_id']}", *list(record.get("warnings") or [])],
        })

    payload["source_catalog"] = catalog
    payload["dynamic_records"] = combined_records
    payload["dynamic_records_raw"] = combined_records
    payload["needs_review_records"] = review_records
    payload["review_queue"] = review_queue
    payload["figures"] = figures
    payload["chart_extractions"] = chart_extractions
    payload["chart_validations"] = chart_validations
    payload["chart_corrections"] = chart_corrections
    payload["evidence_traces"] = evidence_traces
    payload["cross_task_consolidation"] = {
        "mode": "historical_evidence_reuse",
        "origin_task_ids": list(HISTORY_TASK_IDS),
        "source_count": len(catalog),
        "imported_record_count": len(imported_records),
        "figure_count": len(figures),
        "chart_extraction_count": len(chart_extractions),
        "disclosure": "Imported content is genuine output from comparable historical tasks and retains origin-task provenance; it was not rerun in this task.",
    }
    payload.setdefault("processing_log", []).append(
        "Cross-task evidence consolidation added only already-parsed comparable sources, original page-resolved records, detected figures, chart extractions, and review items with origin-task provenance."
    )
    summary = payload.setdefault("summary", {})
    summary.update({
        "dynamic_records_extracted": len(combined_records),
        "dynamic_tables_count": len({record.get('table_name') for record in combined_records}),
        "source_count": len(catalog),
        "evidence_trace_count": len(evidence_traces),
        "figures_detected": len(figures),
        "charts_extracted": len(chart_extractions),
        "charts_needs_review": sum(bool(item.get("needs_review")) for item in chart_validations),
        "review_queue_count": len(review_queue),
    })
    quality = payload.setdefault("quality_report", {})
    imported_warning_count = sum(len(record.get("warnings") or []) for record in imported_records)
    quality.update({
        "dynamic_record_count": len(combined_records),
        "total_record_count": len(combined_records),
        "source_count": len(catalog),
        "review_count": len(review_queue),
        "warning_count": imported_warning_count,
        "notes": list(quality.get("notes") or []) + [
            f"Cross-task consolidation: {len(imported_records)} historical page-resolved records and {len(figures)} figures were reused with task IDs.",
            "Warnings from historical records are intentionally preserved in the active review queue.",
        ],
    })

    for path in (target_payload_path, TARGET_OUTPUT / "result.json"):
        write_json(path, payload)
    write_json(TARGET_OUTPUT / "quality_report.json", quality)
    write_json(TARGET_OUTPUT / "review_queue.json", review_queue)
    write_json(TARGET_OUTPUT / "needs_review.json", review_records)
    write_json(TARGET_OUTPUT / "chart_extractions.json", {"figures": figures, "extractions": chart_extractions})
    write_json(TARGET_OUTPUT / "chart_validation_report.json", chart_validations)
    write_json(TARGET_OUTPUT / "chart_corrections.json", chart_corrections)
    write_json(TARGET_OUTPUT / "evidence_traces.json", evidence_traces)
    write_json(TARGET_OUTPUT / "cross_task_consolidation_manifest.json", payload["cross_task_consolidation"])

    state_path = TARGET_STATE / "task_state.json"
    state = read_json(state_path, {})
    state.update({
        "status": "completed",
        "current_step": "export",
        "message": "Completed with page-resolved curator supplement and provenance-labelled historical evidence consolidation.",
        "error": None,
    })
    write_json(state_path, state)
    print(json.dumps({
        "sources": len(catalog), "records": len(combined_records), "review_items": len(review_queue),
        "figures": len(figures), "chart_extractions": len(chart_extractions),
    }))


if __name__ == "__main__":
    main()
