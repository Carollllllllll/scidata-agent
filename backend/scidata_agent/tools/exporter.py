from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scidata_agent.agent.schemas import AgentState, ExportFiles, DynamicRecord
from scidata_agent.tools.evidence import build_evidence_traces, evidence_trace_rows
from scidata_agent.tools.source_catalog import build_source_catalog, source_catalog_rows, source_catalog_summary


CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _sanitize_csv_value(value):
    """Keep untrusted text inert when a CSV is opened in a spreadsheet."""

    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def _write_csv(data, path: Path) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame = frame.map(_sanitize_csv_value)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def export_results(state: AgentState) -> ExportFiles:
    task_dir = state.output_dir / state.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    state.source_catalog = build_source_catalog(state)
    state.evidence_traces = build_evidence_traces(state)
    catalog_rows = source_catalog_rows(state.source_catalog)
    catalog_summary = source_catalog_summary(state.source_catalog)
    failed_source_count = sum(1 for entry in state.source_catalog if entry.status == "failed")
    state.processing_log.append(
        "Source catalog built: "
        f"sources={catalog_summary['source_catalog_count']}, "
        f"artifacts={catalog_summary['source_artifacts_count']}, "
        f"failed_sources={failed_source_count}."
    )

    record_dicts = [record.model_dump(mode="json") for record in state.final_records]
    csv_path = task_dir / "result.csv"
    json_path = task_dir / "result.json"
    log_path = task_dir / "processing_log.json"
    quality_path = task_dir / "quality_report.json"
    coverage_path = task_dir / "coverage_report.json"
    discovery_path = task_dir / "source_discovery_plan.json"
    arxiv_search_plan_path = task_dir / "arxiv_search_plan.json"
    multi_source_search_plan_path = task_dir / "multi_source_search_plan.json"
    connector_status_csv_path = task_dir / "connector_status.csv"
    connector_status_json_path = task_dir / "connector_status.json"
    discovered_sources_csv_path = task_dir / "discovered_sources.csv"
    discovered_sources_json_path = task_dir / "discovered_sources.json"
    source_selection_plan_path = task_dir / "source_selection_plan.json"
    source_selection_csv_path = task_dir / "source_selection.csv"
    source_triage_csv_path = task_dir / "source_triage.csv"
    source_triage_json_path = task_dir / "source_triage.json"
    source_research_csv_path = task_dir / "source_research.csv"
    source_research_json_path = task_dir / "source_research.json"
    source_catalog_json_path = task_dir / "source_catalog.json"
    source_catalog_csv_path = task_dir / "source_catalog.csv"
    evidence_traces_json_path = task_dir / "evidence_traces.json"
    evidence_traces_csv_path = task_dir / "evidence_traces.csv"
    artifact_action_plan_path = task_dir / "artifact_action_plan.json"
    artifact_action_results_path = task_dir / "artifact_action_results.json"
    artifact_action_history_path = task_dir / "artifact_action_history.json"
    section_plan_path = task_dir / "section_plan.json"
    paper_survey_csv_path = task_dir / "paper_survey.csv"
    paper_survey_json_path = task_dir / "paper_survey.json"
    dynamic_schema_path = task_dir / "dynamic_schema.json"
    dynamic_records_path = task_dir / "dynamic_records.json"
    clean_dynamic_records_path = task_dir / "dynamic_records_clean.json"
    raw_dynamic_records_path = task_dir / "dynamic_records_raw.json"
    needs_review_csv_path = task_dir / "needs_review.csv"
    needs_review_json_path = task_dir / "needs_review.json"
    review_queue_json_path = task_dir / "review_queue.json"
    dynamic_tables_dir = task_dir / "tables"
    chart_extractions_path = task_dir / "chart_extractions.json"
    chart_validation_path = task_dir / "chart_validation_report.json"
    cross_modal_validation_path = task_dir / "cross_modal_validation.json"
    chart_tables_dir = task_dir / "chart_data"
    figures_dir = task_dir / "figures"
    summary_json_path = task_dir / "summary.json"
    final_report_path = task_dir / "final_report.md"
    chart_corrections_path = task_dir / "chart_corrections.json"
    agent_trace_path = task_dir / "agent_trace.json"
    decision_history_path = task_dir / "decision_history.json"
    tool_history_path = task_dir / "tool_history.json"

    _write_csv(record_dicts, csv_path)
    summary_payload = _summary_payload(state)
    raw_dynamic_record_dicts = [record.model_dump(mode="json") for record in state.dynamic_records]
    clean_dynamic_records = state.clean_dynamic_records or state.dynamic_records
    clean_dynamic_record_dicts = [record.model_dump(mode="json") for record in clean_dynamic_records]
    needs_review_dicts = [record.model_dump(mode="json") for record in state.needs_review_records]
    json_payload = {
        "task_id": state.task_id,
        "status": "completed",
        "research_question": state.research_question,
        "summary": summary_payload,
        "task_plan": state.task_plan.model_dump(mode="json") if state.task_plan else None,
        "source_discovery_plan": state.source_discovery_plan.model_dump(mode="json")
        if state.source_discovery_plan
        else None,
        "arxiv_search_plan": state.arxiv_search_plan.model_dump(mode="json") if state.arxiv_search_plan else None,
        "multi_source_search_plan": state.multi_source_search_plan.model_dump(mode="json")
        if state.multi_source_search_plan
        else None,
        "connector_status": state.connector_status,
        "discovered_sources": _discovered_source_rows(state),
        "source_selection_plan": state.source_selection_plan.model_dump(mode="json")
        if state.source_selection_plan
        else None,
        "source_selection_decisions": _source_selection_rows(state),
        "source_triage_decisions": _source_triage_rows(state),
        "source_insights": _source_insight_rows(state),
        "source_catalog": [entry.model_dump(mode="json") for entry in state.source_catalog],
        "evidence_traces": [trace.model_dump(mode="json") for trace in state.evidence_traces],
        "artifact_action_plan": state.artifact_action_plan.model_dump(mode="json")
        if state.artifact_action_plan
        else None,
        "artifact_action_results": [
            result.model_dump(mode="json") for result in state.artifact_action_results
        ],
        "artifact_action_history": [
            iteration.model_dump(mode="json") for iteration in state.artifact_action_history
        ],
        "dynamic_extraction_plan": state.dynamic_extraction_plan.model_dump(mode="json")
        if state.dynamic_extraction_plan
        else None,
        "records": record_dicts,
        "dynamic_records": clean_dynamic_record_dicts,
        "dynamic_records_raw": raw_dynamic_record_dicts,
        "needs_review_records": needs_review_dicts,
        "review_queue": [item.model_dump(mode="json") for item in state.review_queue],
        "sources": [source.model_dump(mode="json") for source in state.sources],
        "quality_report": state.quality_report.model_dump(mode="json"),
        "coverage_report": state.coverage_report.model_dump(mode="json"),
        "cross_modal_checks": [check.model_dump(mode="json") for check in state.cross_modal_checks],
        "chart_corrections": [correction.model_dump(mode="json") for correction in state.chart_corrections],
        "runtime": {
            "iteration": state.runtime_iteration,
            "status": state.runtime_status,
            "stop_reason": state.runtime_stop_reason,
            "decision_count": len(state.agent_decision_history),
            "tool_result_count": len(state.tool_result_history),
            "trace_count": len(state.agent_trace),
        },
        "processing_log": state.processing_log,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    agent_trace_path.write_text(json.dumps(state.agent_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_history_path.write_text(
        json.dumps(state.agent_decision_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tool_history_path.write_text(
        json.dumps(state.tool_result_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path.write_text(json.dumps(state.processing_log, ensure_ascii=False, indent=2), encoding="utf-8")
    quality_path.write_text(
        json.dumps(state.quality_report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    coverage_path.write_text(
        json.dumps(state.coverage_report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    discovery_payload = (
        state.source_discovery_plan.model_dump(mode="json") if state.source_discovery_plan else None
    )
    discovery_path.write_text(json.dumps(discovery_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    arxiv_search_plan_payload = state.arxiv_search_plan.model_dump(mode="json") if state.arxiv_search_plan else None
    arxiv_search_plan_path.write_text(
        json.dumps(arxiv_search_plan_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    multi_source_search_plan_payload = (
        state.multi_source_search_plan.model_dump(mode="json") if state.multi_source_search_plan else None
    )
    multi_source_search_plan_path.write_text(
        json.dumps(multi_source_search_plan_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    connector_status_rows = _connector_status_rows(state)
    _write_csv(connector_status_rows, connector_status_csv_path)
    connector_status_json_path.write_text(
        json.dumps(connector_status_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    discovered_source_rows = _discovered_source_rows(state)
    _write_csv(discovered_source_rows, discovered_sources_csv_path)
    discovered_sources_json_path.write_text(
        json.dumps(discovered_source_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_selection_payload = state.source_selection_plan.model_dump(mode="json") if state.source_selection_plan else None
    source_selection_plan_path.write_text(
        json.dumps(source_selection_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_selection_rows = _source_selection_rows(state)
    _write_csv(source_selection_rows, source_selection_csv_path)
    source_triage_rows = _source_triage_rows(state)
    _write_csv(source_triage_rows, source_triage_csv_path)
    source_triage_json_path.write_text(
        json.dumps(source_triage_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_insight_rows = _source_insight_rows(state)
    _write_csv(source_insight_rows, source_research_csv_path)
    source_research_json_path.write_text(
        json.dumps(source_insight_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_catalog_json_path.write_text(
        json.dumps(
            [entry.model_dump(mode="json") for entry in state.source_catalog],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(catalog_rows, source_catalog_csv_path)
    evidence_traces_json_path.write_text(
        json.dumps(
            [trace.model_dump(mode="json") for trace in state.evidence_traces],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(evidence_trace_rows(state.evidence_traces), evidence_traces_csv_path)
    artifact_action_plan_path.write_text(
        json.dumps(
            state.artifact_action_plan.model_dump(mode="json") if state.artifact_action_plan else None,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_action_results_path.write_text(
        json.dumps(
            [result.model_dump(mode="json") for result in state.artifact_action_results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_action_history_path.write_text(
        json.dumps(
            [iteration.model_dump(mode="json") for iteration in state.artifact_action_history],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    section_plan_path.write_text(json.dumps(_section_payload(state), ensure_ascii=False, indent=2), encoding="utf-8")
    paper_survey_records = build_paper_survey_records(state)
    _write_csv(paper_survey_records, paper_survey_csv_path)
    paper_survey_json_path.write_text(
        json.dumps(paper_survey_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    dynamic_schema_payload = state.dynamic_extraction_plan.model_dump(mode="json") if state.dynamic_extraction_plan else None
    dynamic_schema_path.write_text(json.dumps(dynamic_schema_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dynamic_records_path.write_text(json.dumps(clean_dynamic_record_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_dynamic_records_path.write_text(json.dumps(clean_dynamic_record_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_dynamic_records_path.write_text(json.dumps(raw_dynamic_record_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    needs_review_json_path.write_text(json.dumps(needs_review_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(_dynamic_records_to_rows(state.needs_review_records), needs_review_csv_path)
    review_queue_json_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in state.review_queue], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_dynamic_tables(state, dynamic_tables_dir, records=clean_dynamic_records)
    export_chart_data(state, chart_tables_dir)
    chart_extractions_path.write_text(
        json.dumps(
            {
                "figures": [
                    asset.model_dump(mode="json") for asset in state.parsed_sources.figure_assets
                ],
                "extractions": [
                    extraction.model_dump(mode="json") for extraction in state.chart_extractions
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    chart_validation_path.write_text(
        json.dumps(
            [validation.model_dump(mode="json") for validation in state.chart_validations],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    chart_corrections_path.write_text(
        json.dumps(
            [correction.model_dump(mode="json") for correction in state.chart_corrections],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cross_modal_validation_path.write_text(
        json.dumps(
            [check.model_dump(mode="json") for check in state.cross_modal_checks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(build_human_summary(state, paper_survey_records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    final_report_path.write_text(build_final_report(state, paper_survey_records), encoding="utf-8")

    return ExportFiles(
        csv=str(csv_path),
        json_file=str(json_path),
        processing_log=str(log_path),
        quality_report=str(quality_path),
        coverage_report=str(coverage_path),
        source_discovery_plan=str(discovery_path),
        arxiv_search_plan=str(arxiv_search_plan_path),
        multi_source_search_plan=str(multi_source_search_plan_path),
        connector_status_csv=str(connector_status_csv_path),
        connector_status_json=str(connector_status_json_path),
        source_selection_plan=str(source_selection_plan_path),
        source_selection_csv=str(source_selection_csv_path),
        discovered_sources_csv=str(discovered_sources_csv_path),
        discovered_sources_json=str(discovered_sources_json_path),
        source_triage_csv=str(source_triage_csv_path),
        source_triage_json=str(source_triage_json_path),
        source_research_csv=str(source_research_csv_path),
        source_research_json=str(source_research_json_path),
        source_catalog_json=str(source_catalog_json_path),
        source_catalog_csv=str(source_catalog_csv_path),
        evidence_traces_json=str(evidence_traces_json_path),
        evidence_traces_csv=str(evidence_traces_csv_path),
        artifact_action_plan_json=str(artifact_action_plan_path),
        artifact_action_results_json=str(artifact_action_results_path),
        artifact_action_history_json=str(artifact_action_history_path),
        section_plan=str(section_plan_path),
        monitor_log=str(state.monitor_log_path) if state.monitor_log_path else None,
        paper_survey_csv=str(paper_survey_csv_path),
        paper_survey_json=str(paper_survey_json_path),
        dynamic_schema=str(dynamic_schema_path),
        dynamic_records=str(dynamic_records_path),
        clean_dynamic_records=str(clean_dynamic_records_path),
        needs_review=str(needs_review_csv_path),
        review_queue_json=str(review_queue_json_path),
        dynamic_tables_dir=str(dynamic_tables_dir),
        figures_dir=str(figures_dir) if figures_dir.exists() else None,
        chart_extractions_json=str(chart_extractions_path),
        chart_validation_json=str(chart_validation_path),
        chart_corrections_json=str(chart_corrections_path),
        cross_modal_validation_json=str(cross_modal_validation_path),
        chart_tables_dir=str(chart_tables_dir) if chart_tables_dir.exists() else None,
        final_report=str(final_report_path),
        summary_json=str(summary_json_path),
        agent_trace_json=str(agent_trace_path),
        decision_history_json=str(decision_history_path),
        tool_history_json=str(tool_history_path),
    )


def _summary_payload(state: AgentState) -> dict:
    warnings = state.quality_report.warning_count + state.quality_report.error_count
    return {
        "files_processed": len(state.files),
        "text_blocks_processed": len(state.parsed_sources.text_blocks),
        "heading_candidates_extracted": len(state.parsed_sources.heading_candidates),
        "section_blocks_processed": len(state.parsed_sources.section_blocks),
        "tables_processed": len(state.parsed_sources.tables),
        "records_extracted": len(state.candidate_records),
        "records_after_cleaning": len(state.final_records),
        "dynamic_records_extracted": len(state.clean_dynamic_records or state.dynamic_records),
        "dynamic_tables_count": len({record.table_name for record in (state.clean_dynamic_records or state.dynamic_records)}),
        "discovered_sources_count": len(state.source_discovery_plan.candidate_sources) if state.source_discovery_plan else 0,
        "multi_source_requests_count": len(state.multi_source_search_plan.search_requests) if state.multi_source_search_plan else 0,
        "source_selection_decisions_count": len(state.source_selection_plan.decisions) if state.source_selection_plan else 0,
        "source_selection_selected_count": sum(
            1
            for decision in state.source_selection_plan.decisions
            if decision.decision not in {"reject", "metadata_only"}
        )
        if state.source_selection_plan
        else 0,
        "source_selection_rejected_count": sum(
            1 for decision in state.source_selection_plan.decisions if decision.decision == "reject"
        )
        if state.source_selection_plan
        else 0,
        "source_triage_decisions_count": len(state.source_triage_decisions),
        "source_triage_ingest_selected": sum(1 for decision in state.source_triage_decisions if decision.should_ingest),
        "source_insights_count": len(state.source_insights),
        "connector_status_count": len(state.connector_status),
        "connector_failed_count": sum(1 for item in state.connector_status if item.get("status") == "failed"),
        "downloaded_pdf_count": _downloaded_pdf_count(state),
        "parseable_downloaded_file_count": _parseable_downloaded_file_count(state),
        "figures_detected": len(state.parsed_sources.figure_assets),
        "charts_extracted": len(state.chart_extractions),
        "charts_needs_review": sum(1 for validation in state.chart_validations if validation.needs_review),
        "workflow_alerts": _workflow_alerts(state),
        "skipped_llm_blocks": _skipped_block_count(state.processing_log),
        "warnings": warnings,
    }


def _section_payload(state: AgentState) -> dict:
    return {
        "heading_candidates": [
            candidate.model_dump(mode="json") for candidate in state.parsed_sources.heading_candidates
        ],
        "section_plan": state.parsed_sources.section_plan.model_dump(mode="json")
        if state.parsed_sources.section_plan
        else None,
        "section_blocks": [
            {
                "source_file": block.source_file,
                "section_title": block.section_title,
                "section_type": block.section_type,
                "page_start": block.page_start,
                "page_end": block.page_end,
                "chunk_id": block.chunk_id,
                "confidence": block.confidence,
                "chars": len(block.text),
                "preview": block.text[:1200],
                "raw": block.raw,
            }
            for block in state.parsed_sources.section_blocks
        ],
    }


def _discovered_source_rows(state: AgentState) -> list[dict]:
    if not state.source_discovery_plan:
        return []
    rows = []
    for source in state.source_discovery_plan.candidate_sources:
        metadata = source.metadata or {}
        rows.append(
            {
                "source_id": source.source_id,
                "source_cluster_id": source.source_cluster_id,
                "title": source.title,
                "source_type": source.source_type,
                "provider": metadata.get("provider"),
                "url": source.url,
                "query": source.query,
                "reason": source.reason,
                "description": source.description,
                "confidence": source.confidence,
                "doi": metadata.get("doi") or metadata.get("DOI"),
                "pdf_url": metadata.get("pdf_url"),
                "open_access_url": metadata.get("open_access_url"),
                "downloaded_path": metadata.get("downloaded_path"),
                "published": metadata.get("published") or metadata.get("publication_date") or metadata.get("published_date"),
                "year": metadata.get("year") or metadata.get("publication_year"),
                "authors": _plain(metadata.get("authors") or metadata.get("creators")),
                "venue": metadata.get("venue"),
                "providers": _plain(metadata.get("providers")),
                "source_ids": _plain(metadata.get("source_ids")),
                "alternate_urls": _plain(metadata.get("alternate_urls")),
                "source_conflicts": json.dumps(metadata.get("source_conflicts", []), ensure_ascii=False),
                "metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    return rows


def _connector_status_rows(state: AgentState) -> list[dict]:
    rows = []
    for index, item in enumerate(state.connector_status, start=1):
        rows.append(
            {
                "index": index,
                "connector": item.get("connector"),
                "query": item.get("query"),
                "status": item.get("status"),
                "added": item.get("added"),
                "error": item.get("error"),
                "metadata": json.dumps(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"connector", "query", "status", "added", "error"}
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _source_selection_rows(state: AgentState) -> list[dict]:
    if not state.source_discovery_plan or not state.source_selection_plan:
        return []
    sources = {source.source_id: source for source in state.source_discovery_plan.candidate_sources}
    rows = []
    for decision in state.source_selection_plan.decisions:
        source = sources.get(decision.source_id)
        metadata = source.metadata if source else {}
        rows.append(
            {
                "source_id": decision.source_id,
                "title": source.title if source else None,
                "provider": metadata.get("provider"),
                "source_type": source.source_type if source else None,
                "published": metadata.get("published") or metadata.get("publication_date") or metadata.get("published_date"),
                "year": metadata.get("year") or metadata.get("publication_year"),
                "decision": decision.decision,
                "priority": decision.priority,
                "source_role": decision.source_role,
                "priority_score": decision.priority_score,
                "reason": decision.reason,
                "matched_requirements": _plain(decision.matched_requirements),
                "expected_extractable_fields": _plain(decision.expected_extractable_fields),
                "risk_notes": _plain(decision.risk_notes),
                "url": source.url if source else None,
                "query": source.query if source else None,
                "pdf_url": metadata.get("pdf_url"),
                "doi": metadata.get("doi") or metadata.get("DOI"),
            }
        )
    return rows


def _source_triage_rows(state: AgentState) -> list[dict]:
    rows = []
    for decision in state.source_triage_decisions:
        metadata = decision.metadata or {}
        rows.append(
            {
                "source_id": decision.source_id,
                "title": decision.title,
                "provider": decision.provider,
                "source_type": decision.source_type,
                "relevance_score": decision.relevance_score,
                "recommended_action": decision.recommended_action,
                "should_ingest": decision.should_ingest,
                "reason": decision.reason,
                "estimated_download_size": decision.estimated_download_size,
                "estimated_cost": decision.estimated_cost,
                "risk": decision.risk,
                "url": metadata.get("url"),
                "query": metadata.get("query"),
                "doi": metadata.get("doi"),
                "pdf_url": metadata.get("pdf_url"),
                "open_access_url": metadata.get("open_access_url"),
                "metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    return rows


def _source_insight_rows(state: AgentState) -> list[dict]:
    rows = []
    for insight in state.source_insights:
        rows.append(
            {
                "insight_id": insight.insight_id,
                "source_id": insight.source_id,
                "title": insight.title,
                "provider": insight.provider,
                "source_type": insight.source_type,
                "insight_type": insight.insight_type,
                "url": insight.url,
                "confidence": insight.confidence,
                "content_chars": len(insight.content),
                "content_preview": insight.content[:2000],
                "content": insight.content,
                "metadata": json.dumps(insight.metadata, ensure_ascii=False),
            }
        )
    return rows


def export_dynamic_tables(state: AgentState, output_dir: Path, records: list[DynamicRecord] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for record in records if records is not None else state.dynamic_records:
        row = _dynamic_record_to_row(record)
        grouped.setdefault(record.table_name, []).append(row)

    table_names = set(grouped)
    if state.dynamic_extraction_plan:
        table_names.update(table.table_name for table in state.dynamic_extraction_plan.dynamic_tables)
    for table_name in sorted(table_names):
        rows = grouped.get(table_name, [])
        path = output_dir / f"{_safe_filename(table_name)}.csv"
        if rows:
            _write_csv(rows, path)
        else:
            columns = _dynamic_table_columns(state, table_name)
            _write_csv(pd.DataFrame(columns=columns), path)


def export_chart_data(state: AgentState, output_dir: Path) -> None:
    """Write one long-format CSV per extracted chart plus an index CSV."""
    if not state.chart_extractions:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_by_id = {asset.figure_id: asset for asset in state.parsed_sources.figure_assets}
    index_rows: list[dict] = []
    for extraction in state.chart_extractions:
        figure = figures_by_id.get(extraction.figure_id)
        rows = []
        for series in extraction.series:
            for point in series.points:
                rows.append(
                    {
                        "figure_label": figure.label if figure else None,
                        "series_name": series.name,
                        "x": point[0],
                        "y": point[1],
                        "x_unit": extraction.x_axis.unit,
                        "y_unit": extraction.y_axis.unit,
                        "approximate": extraction.approximate,
                    }
                )
        if not rows:
            continue
        csv_name = f"chart_data_{_safe_filename(extraction.extraction_id)}.csv"
        _write_csv(rows, output_dir / csv_name)
        index_rows.append(
            {
                "extraction_id": extraction.extraction_id,
                "figure_id": extraction.figure_id,
                "figure_label": figure.label if figure else None,
                "source_file": extraction.source_file,
                "page": extraction.page,
                "chart_type": extraction.chart_type,
                "series_count": len(extraction.series),
                "point_count": sum(len(series.points) for series in extraction.series),
                "x_axis_label": extraction.x_axis.label,
                "x_axis_unit": extraction.x_axis.unit,
                "x_axis_scale": extraction.x_axis.scale,
                "y_axis_label": extraction.y_axis.label,
                "y_axis_unit": extraction.y_axis.unit,
                "y_axis_scale": extraction.y_axis.scale,
                "confidence": extraction.confidence,
                "approximate": extraction.approximate,
                "caption": figure.caption if figure else None,
                "image_path": figure.image_path if figure else None,
                "data_csv": csv_name,
            }
        )
    if index_rows:
        _write_csv(index_rows, output_dir / "chart_data_index.csv")


def _dynamic_table_columns(state: AgentState, table_name: str) -> list[str]:
    columns: list[str] = []
    if state.dynamic_extraction_plan:
        for table in state.dynamic_extraction_plan.dynamic_tables:
            if table.table_name == table_name:
                columns = [field.name for field in table.fields]
                break
    return columns + [
        "record_id",
        "source_file",
        "source_type",
        "page",
        "section_title",
        "section_type",
        "page_start",
        "page_end",
        "evidence_text",
        "confidence",
        "warnings",
    ]


def _dynamic_records_to_rows(records: list[DynamicRecord]) -> list[dict]:
    return [_dynamic_record_to_row(record) for record in records]


def _dynamic_record_to_row(record: DynamicRecord) -> dict:
    row = dict(record.fields)
    row.update(
        {
            "record_id": record.record_id,
            "table_name": record.table_name,
            "source_file": record.source_file,
            "source_type": record.source_type.value,
            "page": record.page,
            "section_title": record.raw.get("section_title"),
            "section_type": record.raw.get("section_type"),
            "page_start": record.raw.get("page_start"),
            "page_end": record.raw.get("page_end"),
            "evidence_text": record.evidence_text,
            "confidence": record.confidence,
            "warnings": "; ".join(record.warnings),
            "needs_review_reasons": "; ".join(record.raw.get("needs_review_reasons", [])),
        }
    )
    return row


def build_human_summary(state: AgentState, paper_survey_records: list[dict]) -> dict:
    dynamic_records = state.clean_dynamic_records or state.dynamic_records
    dynamic_tables = sorted({record.table_name for record in dynamic_records})
    runtime_status = str(state.runtime_status or "")
    result_status = (
        "partial"
        if runtime_status in {"running", "partial", "failed"}
        else "completed"
    )
    return {
        "task_id": state.task_id,
        "research_question": state.research_question,
        "status": result_status,
        "runtime_status": runtime_status,
        "runtime_iteration": state.runtime_iteration,
        "runtime_stop_reason": state.runtime_stop_reason,
        "stop_rejections_count": len(state.stop_rejections),
        "files_processed": len(state.files),
        "heading_candidates": len(state.parsed_sources.heading_candidates),
        "section_blocks": len(state.parsed_sources.section_blocks),
        "section_types": sorted({block.section_type for block in state.parsed_sources.section_blocks}),
        "papers_summarized": len(paper_survey_records),
        "scientific_records": len(state.final_records),
        "dynamic_records": len(dynamic_records),
        "dynamic_records_raw": len(state.dynamic_records),
        "needs_review_records": len(state.needs_review_records),
        "discovered_sources": len(state.source_discovery_plan.candidate_sources) if state.source_discovery_plan else 0,
        "multi_source_requests": len(state.multi_source_search_plan.search_requests) if state.multi_source_search_plan else 0,
        "source_selection_decisions": len(state.source_selection_plan.decisions) if state.source_selection_plan else 0,
        "source_selection_selected": sum(
            1
            for decision in state.source_selection_plan.decisions
            if decision.decision not in {"reject", "metadata_only"}
        )
        if state.source_selection_plan
        else 0,
        "source_triage_decisions": len(state.source_triage_decisions),
        "source_triage_ingest_selected": sum(1 for decision in state.source_triage_decisions if decision.should_ingest),
        "source_insights": len(state.source_insights),
        "connector_status": {
            "checked": len(state.connector_status),
            "failed": sum(1 for item in state.connector_status if item.get("status") == "failed"),
            "completed": sum(1 for item in state.connector_status if item.get("status") == "completed"),
        },
        "downloaded_pdf_count": _downloaded_pdf_count(state),
        "parseable_downloaded_file_count": _parseable_downloaded_file_count(state),
        "figures_detected": len(state.parsed_sources.figure_assets),
        "charts_extracted": len(state.chart_extractions),
        "charts_needs_review": sum(1 for validation in state.chart_validations if validation.needs_review),
        "workflow_alerts": _workflow_alerts(state),
        "dynamic_tables": dynamic_tables,
        "quality": {
            "issues": state.quality_report.issue_count,
            "warnings": state.quality_report.warning_count,
            "errors": state.quality_report.error_count,
            "evidence_coverage": state.quality_report.evidence_coverage,
            "value_evidence_coverage": state.quality_report.value_evidence_coverage,
        },
        "recommended_files": {
            "human_report": "final_report.md",
            "paper_survey": "paper_survey.csv",
            "dynamic_schema": "dynamic_schema.json",
            "multi_source_search_plan": "multi_source_search_plan.json",
            "connector_status": "connector_status.csv",
            "discovered_sources": "discovered_sources.csv",
            "source_selection": "source_selection.csv",
            "source_selection_plan": "source_selection_plan.json",
            "source_triage": "source_triage.csv",
            "source_research": "source_research.csv",
            "arxiv_search_plan": "arxiv_search_plan.json",
            "dynamic_tables": "tables/",
            "figures": "figures/",
            "chart_extractions": "chart_extractions.json",
            "chart_validation_report": "chart_validation_report.json",
            "chart_data": "chart_data/",
            "dynamic_records_raw": "dynamic_records_raw.json",
            "dynamic_records_clean": "dynamic_records_clean.json",
            "needs_review": "needs_review.csv",
            "metric_records": "result.csv",
            "quality_report": "quality_report.json",
            "section_plan": "section_plan.json",
        },
    }


def build_final_report(state: AgentState, paper_survey_records: list[dict]) -> str:
    summary = _summary_payload(state)
    lines = [
        f"# SciData Agent Final Report",
        "",
        f"## Task",
        "",
        f"- Task ID: `{state.task_id}`",
        f"- Research question: {state.research_question}",
        f"- Files processed: {summary['files_processed']}",
        f"- Text blocks processed: {summary['text_blocks_processed']}",
        f"- Heading candidates extracted: {summary['heading_candidates_extracted']}",
        f"- Section blocks processed: {summary['section_blocks_processed']}",
        f"- Discovered sources: {summary['discovered_sources_count']}",
        f"- Multi-source search requests: {summary['multi_source_requests_count']}",
        f"- LLM source selection decisions: {summary['source_selection_decisions_count']}",
        f"- LLM-selected sources before safety triage: {summary['source_selection_selected_count']}",
        f"- Source triage decisions: {summary['source_triage_decisions_count']}",
        f"- Sources selected for ingestion: {summary['source_triage_ingest_selected']}",
        f"- Source insights: {summary['source_insights_count']}",
        f"- Connector checks: {summary['connector_status_count']} / failed: {summary['connector_failed_count']}",
        f"- Downloaded PDFs: {summary['downloaded_pdf_count']}",
        f"- Parseable downloaded files: {summary['parseable_downloaded_file_count']}",
        f"- Metric records extracted: {summary['records_after_cleaning']}",
        f"- Dynamic records after cleaning: {summary['dynamic_records_extracted']}",
        f"- Dynamic records raw: {len(state.dynamic_records)}",
        f"- Needs review records: {len(state.needs_review_records)}",
        f"- Dynamic tables with data: {summary['dynamic_tables_count']}",
        "",
    ]
    if summary["workflow_alerts"]:
        lines.extend(["## Run Alerts", ""])
        for alert in summary["workflow_alerts"]:
            lines.append(f"- {alert}")
        lines.append("")

    if state.dynamic_extraction_plan:
        lines.extend(
            [
                "## Dynamic Extraction Schema",
                "",
                f"- Domain: {state.dynamic_extraction_plan.domain}",
                f"- Task type: {state.dynamic_extraction_plan.task_type}",
                f"- User focus: {', '.join(state.dynamic_extraction_plan.user_focus) if state.dynamic_extraction_plan.user_focus else 'not specified'}",
                "",
                "| Table | Entity | Fields |",
                "|---|---|---|",
            ]
        )
        for table in state.dynamic_extraction_plan.dynamic_tables:
            fields = ", ".join(field.name for field in table.fields)
            lines.append(f"| {table.table_name} | {table.entity_type} | {fields} |")
        lines.append("")

    if state.connector_status:
        lines.extend(
            [
                "## Connector Status",
                "",
                f"- Checked: {len(state.connector_status)}",
                f"- Failed: {sum(1 for item in state.connector_status if item.get('status') == 'failed')}",
                "",
                "| Connector | Status | Added | Query | Error |",
                "|---|---|---|---|---|",
            ]
        )
        for row in _connector_status_rows(state)[:20]:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(row.get(key))
                    for key in ["connector", "status", "added", "query", "error"]
                )
                + " |"
            )
        lines.append("")

    if state.source_discovery_plan:
        providers: dict[str, int] = {}
        for source in state.source_discovery_plan.candidate_sources:
            provider = str(source.metadata.get("provider") or source.source_type or "unknown")
            providers[provider] = providers.get(provider, 0) + 1
        lines.extend(
            [
                "## Multi-source Discovery",
                "",
                f"- Search requests: {summary['multi_source_requests_count']}",
                f"- Candidate sources: {summary['discovered_sources_count']}",
                f"- Providers: {', '.join(f'{key}({value})' for key, value in sorted(providers.items())) if providers else 'none'}",
                "",
                "| Provider | Type | Title | URL | Query |",
                "|---|---|---|---|---|",
            ]
        )
        for row in _discovered_source_rows(state)[:20]:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(row.get(key))
                    for key in ["provider", "source_type", "title", "url", "query"]
                )
                + " |"
            )
        lines.append("")

    if state.source_selection_plan:
        decision_counts: dict[str, int] = {}
        priority_counts: dict[str, int] = {}
        for decision in state.source_selection_plan.decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            priority_counts[decision.priority] = priority_counts.get(decision.priority, 0) + 1
        lines.extend(
            [
                "## Source Selection",
                "",
                f"- Decisions: {len(state.source_selection_plan.decisions)}",
                f"- Decision counts: {', '.join(f'{key}({value})' for key, value in sorted(decision_counts.items()))}",
                f"- Priority counts: {', '.join(f'{key}({value})' for key, value in sorted(priority_counts.items()))}",
                f"- Time range interpreted: {state.source_selection_plan.time_range_interpreted or 'not specified'}",
                f"- Summary: {state.source_selection_plan.selection_summary or 'not specified'}",
                "",
                "| Decision | Priority | Role | Provider | Title | Score | Reason |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in _source_selection_rows(state)[:20]:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(row.get(key))
                    for key in ["decision", "priority", "source_role", "provider", "title", "priority_score", "reason"]
                )
                + " |"
            )
        lines.append("")

    if state.source_triage_decisions:
        action_counts: dict[str, int] = {}
        for decision in state.source_triage_decisions:
            action_counts[decision.recommended_action] = action_counts.get(decision.recommended_action, 0) + 1
        lines.extend(
            [
                "## Source Triage",
                "",
                f"- Decisions: {len(state.source_triage_decisions)}",
                f"- Selected for ingestion: {sum(1 for decision in state.source_triage_decisions if decision.should_ingest)}",
                f"- Actions: {', '.join(f'{key}({value})' for key, value in sorted(action_counts.items()))}",
                "",
                "| Action | Provider | Title | Score | Cost | Risk | Reason |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in _source_triage_rows(state)[:20]:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(row.get(key))
                    for key in ["recommended_action", "provider", "title", "relevance_score", "estimated_cost", "risk", "reason"]
                )
                + " |"
            )
        lines.append("")

    if state.source_insights:
        type_counts: dict[str, int] = {}
        provider_counts: dict[str, int] = {}
        for insight in state.source_insights:
            type_counts[insight.insight_type] = type_counts.get(insight.insight_type, 0) + 1
            provider = insight.provider or "unknown"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        lines.extend(
            [
                "## Source Research",
                "",
                f"- Insights: {len(state.source_insights)}",
                f"- Providers: {', '.join(f'{key}({value})' for key, value in sorted(provider_counts.items()))}",
                f"- Insight types: {', '.join(f'{key}({value})' for key, value in sorted(type_counts.items()))}",
                "",
                "| Provider | Type | Title | Insight | Preview |",
                "|---|---|---|---|---|",
            ]
        )
        for row in _source_insight_rows(state)[:20]:
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(row.get(key))
                    for key in ["provider", "source_type", "title", "insight_type", "content_preview"]
                )
                + " |"
            )
        lines.append("")

    if state.parsed_sources.section_blocks:
        section_counts: dict[str, int] = {}
        for block in state.parsed_sources.section_blocks:
            section_counts[block.section_type] = section_counts.get(block.section_type, 0) + 1
        interpreter = "LLM" if state.parsed_sources.section_plan and state.parsed_sources.section_plan.used_llm else "fallback"
        lines.extend(
            [
                "## Paper Structure",
                "",
                f"- Heading candidates: {len(state.parsed_sources.heading_candidates)}",
                f"- Section blocks: {len(state.parsed_sources.section_blocks)}",
                f"- Section interpreter: {interpreter}",
                f"- Section types: {', '.join(f'{key}({value})' for key, value in sorted(section_counts.items()))}",
                f"- Skipped LLM blocks: {summary['skipped_llm_blocks']}",
                "",
                "| Paper | Section | Type | Pages | Confidence |",
                "|---|---|---|---|---|",
            ]
        )
        seen_sections = set()
        for block in state.parsed_sources.section_blocks:
            key = (block.source_file, block.section_title, block.page_start, block.page_end)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            pages = f"{block.page_start}-{block.page_end}" if block.page_start != block.page_end else str(block.page_start)
            lines.append(
                f"| {_md_cell(block.source_file)} | {_md_cell(block.section_title)} | {_md_cell(block.section_type)} | {_md_cell(pages)} | {_md_cell(round(block.confidence, 3))} |"
            )
            if len(seen_sections) >= 12:
                break
        lines.append("")

    lines.extend(["## Paper Survey", "", "| Paper | Methods | Baselines | Datasets / Objects | Metrics | Records |", "|---|---|---|---|---|---|"])
    for row in paper_survey_records[:12]:
        lines.append(
            "| "
            + " | ".join(
                _md_cell(row.get(key))
                for key in ["paper_title", "methods", "baselines", "datasets_or_materials", "metrics", "record_count"]
            )
            + " |"
        )
    lines.append("")

    dynamic_grouped: dict[str, list] = {}
    for record in state.clean_dynamic_records or state.dynamic_records:
        dynamic_grouped.setdefault(record.table_name, []).append(record)
    lines.extend(["## Dynamic Tables Preview", ""])
    if not dynamic_grouped:
        lines.append("No dynamic records were extracted.")
    for table_name, records in sorted(dynamic_grouped.items()):
        lines.extend([f"### {table_name}", ""])
        for record in records[:5]:
            fields_preview = "; ".join(f"{key}={_plain(value)}" for key, value in list(record.fields.items())[:6])
            lines.append(f"- {fields_preview} (source: {record.source_file}, page: {record.page})")
        lines.append("")

    if state.parsed_sources.figure_assets or state.chart_extractions:
        figures_by_id = {asset.figure_id: asset for asset in state.parsed_sources.figure_assets}
        validations_by_id = {validation.figure_id: validation for validation in state.chart_validations}
        lines.extend(
            [
                "## Figure & Chart Extraction",
                "",
                f"- Figures detected: {len(state.parsed_sources.figure_assets)}",
                f"- Charts extracted (Qwen-VL): {len(state.chart_extractions)}",
                f"- Charts needing review: {sum(1 for v in state.chart_validations if v.needs_review)}",
                "",
                "| Figure | Source | Page | Type | Series | Points | Confidence | Validation |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for extraction in state.chart_extractions[:20]:
            figure = figures_by_id.get(extraction.figure_id)
            validation = validations_by_id.get(extraction.figure_id)
            issue_summary = "passed"
            if validation and validation.issues:
                issue_summary = "; ".join(issue.code for issue in validation.issues[:3])
            lines.append(
                "| "
                + " | ".join(
                    _md_cell(value)
                    for value in [
                        figure.label if figure else extraction.figure_id,
                        extraction.source_file,
                        extraction.page,
                        extraction.chart_type,
                        len(extraction.series),
                        sum(len(series.points) for series in extraction.series),
                        round(extraction.confidence, 3),
                        issue_summary,
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Quality",
            "",
            f"- Issues: {state.quality_report.issue_count}",
            f"- Warnings: {state.quality_report.warning_count}",
            f"- Errors: {state.quality_report.error_count}",
            f"- Conflicts: {state.quality_report.conflict_count}",
            f"- Evidence coverage: {state.quality_report.evidence_coverage}",
            f"- Value evidence coverage: {state.quality_report.value_evidence_coverage}",
            "",
            "## Output Guide",
            "",
            "- `tables/*.csv`: dynamic tables generated from the task-specific schema.",
            "- `figures/*.png`: figure regions rendered from PDFs for chart extraction.",
            "- `chart_extractions.json`: VL-extracted chart axes, legends, and data points with figure provenance.",
            "- `chart_data/chart_data_index.csv` + `chart_data/chart_data_*.csv`: long-format chart data points (approximate).",
            "- `chart_validation_report.json`: deterministic axis/series/unit checks for every extracted chart.",
            "- `dynamic_schema.json`: the LLM-generated extraction schema for this task.",
            "- `multi_source_search_plan.json`: the LLM-generated plan for arXiv, OpenAlex, Semantic Scholar, Crossref, Zenodo, Figshare, and GitHub.",
            "- `connector_status.csv`: per-connector success/failure table with query and error messages.",
            "- `discovered_sources.csv`: normalized multi-source search results with provider, URL, query, and metadata.",
            "- `source_selection.csv`: LLM source-selection decisions before executable safety triage.",
            "- `source_selection_plan.json`: full LLM source-selection plan with time range and reasons.",
            "- `source_triage.csv`: source-level ingestion decisions, including why each source was downloaded, deferred, or kept as metadata.",
            "- `source_research.csv`: source-level research evidence collected from metadata, README files, file manifests, and downloaded files.",
            "- `arxiv_search_plan.json`: the LLM-generated arXiv search queries and selection criteria.",
            "- `dynamic_records_clean.json`: cleaned and merged dynamic records with evidence.",
            "- `dynamic_records_raw.json`: raw LLM dynamic records before curation.",
            "- `needs_review.csv`: records or fields flagged by deterministic quality rules.",
            "- `paper_survey.csv`: paper-level survey summary.",
            "- `result.csv`: metric-oriented scientific records.",
            "- `quality_report.json`: evidence, warning, error, and conflict checks.",
            "- `section_plan.json`: heading candidates, LLM section interpretation, and section-aware chunks.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_paper_survey_records(state: AgentState) -> list[dict]:
    arxiv_by_filename = _arxiv_sources_by_filename(state)
    grouped: dict[str, list] = {}
    for record in state.final_records:
        grouped.setdefault(record.source_file, []).append(record)
    dynamic_grouped: dict[str, list[DynamicRecord]] = {}
    for record in state.clean_dynamic_records or state.dynamic_records:
        dynamic_grouped.setdefault(record.source_file, []).append(record)

    rows: list[dict] = []
    for source_file, records in grouped.items():
        arxiv_source = arxiv_by_filename.get(source_file)
        first_record = records[0] if records else None
        paper_title = _first_non_empty(
            arxiv_source.title if arxiv_source else None,
            first_record.paper_title if first_record else None,
        )
        row = {
            "paper_title": paper_title,
            "authors": "; ".join(arxiv_source.metadata.get("authors", [])) if arxiv_source else None,
            "published": arxiv_source.metadata.get("published") if arxiv_source else None,
            "updated": arxiv_source.metadata.get("updated") if arxiv_source else None,
            "arxiv_url": arxiv_source.url if arxiv_source else None,
            "pdf_url": arxiv_source.metadata.get("pdf_url") if arxiv_source else None,
            "downloaded_path": arxiv_source.metadata.get("downloaded_path") if arxiv_source else None,
            "source_file": source_file,
            "methods": _build_paper_method_summary(paper_title, records, dynamic_grouped.get(source_file, [])),
            "baselines": _build_paper_baseline_summary(paper_title, records, dynamic_grouped.get(source_file, [])),
            "datasets_or_materials": _join_unique(record.material for record in records),
            "metrics": _join_unique(record.metric_name for record in records),
            "record_count": len(records),
            "numeric_record_count": sum(1 for record in records if record.metric_value is not None),
            "pages": _join_unique(str(record.page) for record in records if record.page is not None),
            "evidence_samples": " || ".join(
                record.evidence_text or "" for record in records[:5] if record.evidence_text
            ),
        }
        rows.append(row)

    if rows:
        return rows

    if not state.source_discovery_plan:
        return []
    for source in state.source_discovery_plan.candidate_sources:
        if source.source_type != "paper":
            continue
        rows.append(
            {
                "paper_title": source.title,
                "authors": "; ".join(source.metadata.get("authors", [])),
                "published": source.metadata.get("published"),
                "updated": source.metadata.get("updated"),
                "arxiv_url": source.url,
                "pdf_url": source.metadata.get("pdf_url"),
                "downloaded_path": source.metadata.get("downloaded_path"),
                "source_file": None,
                "methods": None,
                "baselines": None,
                "datasets_or_materials": None,
                "metrics": None,
                "record_count": 0,
                "numeric_record_count": 0,
                "pages": None,
                "evidence_samples": None,
            }
        )
    return rows


def _build_paper_method_summary(
    paper_title: str | None,
    records: list,
    dynamic_records: list[DynamicRecord],
) -> str | None:
    summary_items: list[str] = []
    module_items: list[str] = []
    title_method = _method_name_from_title(paper_title)

    if title_method:
        _append_unique(summary_items, title_method)

    for record in dynamic_records:
        table = record.table_name.lower()
        fields = record.fields
        if table in {"method_architecture", "method_details", "methods", "model_architecture"}:
            method_name = _first_meaningful_text(
                fields.get("method_name"),
                fields.get("architecture_name"),
                fields.get("model_name"),
                fields.get("system_name"),
                fields.get("approach_name"),
                fields.get("framework_name"),
            )
            if method_name:
                _append_unique(summary_items, method_name)

            architecture = _first_meaningful_text(
                fields.get("architecture_paradigm"),
                fields.get("architecture"),
                fields.get("method_type"),
                fields.get("paradigm"),
            )
            backbones = _value_items(
                _first_non_empty(
                    fields.get("backbone_networks"),
                    fields.get("backbone"),
                    fields.get("backbones"),
                    fields.get("model_backbone"),
                )
            )
            details = []
            if architecture:
                details.append(architecture)
            if backbones:
                details.append(", ".join(backbones[:4]))
            if details:
                _append_unique(summary_items, " (" + "; ".join(details) + ")" if not title_method and not method_name else "; ".join(details))

        elif table in {"key_modules", "modules", "components"}:
            for value in _value_items(
                _first_non_empty(
                    fields.get("module_name"),
                    fields.get("component_name"),
                    fields.get("key_module"),
                    fields.get("module"),
                )
            ):
                if _is_meaningful_method_text(value):
                    _append_unique(module_items, value)

    if module_items:
        _append_unique(summary_items, "Key modules: " + ", ".join(module_items[:5]))

    fallback_methods = [record.method for record in records if _is_meaningful_method_text(record.method, paper_title=paper_title)]
    for method in fallback_methods:
        _append_unique(summary_items, str(method).strip())

    cleaned = [_clean_method_display(item) for item in summary_items]
    cleaned = [item for item in cleaned if _is_meaningful_method_text(item, paper_title=paper_title)]
    return "; ".join(cleaned[:6]) if cleaned else None


def _build_paper_baseline_summary(
    paper_title: str | None,
    records: list,
    dynamic_records: list[DynamicRecord],
) -> str | None:
    baselines: list[str] = []
    title_method = _method_name_from_title(paper_title)
    for record in dynamic_records:
        fields = record.fields
        for key in [
            "baseline_comparison",
            "baselines",
            "compared_baselines",
            "compared_methods",
            "baseline_methods",
        ]:
            for item in _split_method_like_items(fields.get(key)):
                if _looks_like_baseline(item, title_method):
                    _append_unique_baseline(baselines, item)
    for record in records:
        for item in _split_method_like_items(record.method):
            if _looks_like_baseline(item, title_method):
                _append_unique_baseline(baselines, item)
        raw = getattr(record, "raw", {}) or {}
        attributes = raw.get("attributes") if isinstance(raw, dict) else {}
        if isinstance(attributes, dict):
            for key in ["baseline", "baselines", "compared_methods"]:
                for item in _split_method_like_items(attributes.get(key)):
                    if _looks_like_baseline(item, title_method):
                        _append_unique_baseline(baselines, item)
    return "; ".join(baselines[:10]) if baselines else None


def _method_name_from_title(title: str | None) -> str | None:
    if not title:
        return None
    title = str(title).strip()
    if ":" in title:
        candidate = title.split(":", 1)[0].strip()
        if _looks_like_named_method(candidate):
            return candidate
    return None


def _looks_like_named_method(value: str) -> bool:
    text = value.strip()
    if not _is_meaningful_method_text(text):
        return False
    has_acronym = bool(re.search(r"\b[A-Z][A-Z0-9-]{2,}\b", text))
    has_model_suffix = bool(re.search(r"\b(VTON|Net|Former|BERT|GPT|Diffusion|Bench)\b", text, flags=re.IGNORECASE))
    return len(text) <= 80 and (has_acronym or has_model_suffix or "-" in text)


def _first_meaningful_text(*values) -> str | None:
    for value in values:
        for item in _value_items(value):
            if _is_meaningful_method_text(item):
                return item
    return None


def _value_items(value) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    text = str(value).strip()
    return [text] if text else []


def _is_meaningful_method_text(value, paper_title: str | None = None) -> bool:
    if value in (None, "", []):
        return False
    text = str(value).strip()
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {
        "ours",
        "our",
        "our method",
        "ours method",
        "proposed method",
        "the proposed method",
        "proposed approach",
        "our approach",
        "the method",
        "this method",
        "method",
        "baseline",
        "baseline method",
        "virtual try on method",
        "try on method",
        "not specified",
        "unknown",
        "none",
        "null",
    }:
        return False
    if normalized.startswith("baseline"):
        return False
    if normalized.startswith("ours ") or normalized.startswith("our method "):
        return False
    if " no " in f" {normalized} " and len(normalized.split()) <= 6:
        return False
    if normalized.endswith(" only") and len(normalized.split()) <= 4:
        return False
    title_method = _method_name_from_title(paper_title)
    if _looks_like_baseline(text, title_method):
        return False
    return True


def _clean_method_display(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"^(ours|our method|proposed method|the proposed method)\s*[:,-]\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _append_unique(items: list[str], value: str | None) -> None:
    if not value:
        return
    text = _clean_method_display(value)
    if not _is_meaningful_method_text(text):
        return
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    existing = {re.sub(r"[^a-z0-9]+", "", item.lower()) for item in items}
    if key not in existing:
        items.append(text)


def _append_unique_baseline(items: list[str], value: str | None) -> None:
    if not value:
        return
    text = _clean_method_display(value)
    if not text:
        return
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    existing = {re.sub(r"[^a-z0-9]+", "", item.lower()) for item in items}
    if key not in existing:
        items.append(text)


def _split_method_like_items(value) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r";|,|\band\b|\bvs\.?\b", str(value), flags=re.IGNORECASE)
    return [item.strip() for item in raw_items if item.strip()]


def _looks_like_baseline(value: str, title_method: str | None = None) -> bool:
    text = str(value).strip()
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if title_method and normalized == re.sub(r"[^a-z0-9]+", "", title_method.lower()):
        return False
    if re.search(r"\b(baseline|sota|comparison|compared)\b", text, flags=re.IGNORECASE):
        return True
    # Method lists in metric/result tables usually contain competing methods; keep the paper's own title method out.
    known_vton_baselines = {
        "dcivton",
        "ladivton",
        "stablevton",
        "idmvton",
        "catvton",
        "sdviton",
        "boowvton",
        "gpvton",
        "hrvton",
        "pfafn",
        "cpvton",
    }
    return normalized in known_vton_baselines


def _downloaded_pdf_count(state: AgentState) -> int:
    if not state.source_discovery_plan:
        return 0
    count = 0
    for source in state.source_discovery_plan.candidate_sources:
        downloaded_paths = list(source.metadata.get("downloaded_paths") or [])
        if source.metadata.get("downloaded_path"):
            downloaded_paths.append(source.metadata["downloaded_path"])
        for path in downloaded_paths:
            if str(path).lower().endswith(".pdf"):
                count += 1
    return count


def _parseable_downloaded_file_count(state: AgentState) -> int:
    parseable_suffixes = {".pdf", ".csv", ".tsv", ".xlsx", ".xls"}
    count = 0
    seen = set()
    for uploaded in state.files:
        suffix = Path(uploaded.filename).suffix.lower()
        key = str(uploaded.path)
        if suffix in parseable_suffixes and key not in seen:
            count += 1
            seen.add(key)
    return count


def _workflow_alerts(state: AgentState) -> list[str]:
    alerts: list[str] = []
    failed = [item for item in state.connector_status if item.get("status") == "failed"]
    if failed:
        names = ", ".join(str(item.get("connector")) for item in failed[:8])
        alerts.append(f"Multi-source search was partial: {len(failed)} connector request(s) failed ({names}).")

    is_remote_research = bool(state.multi_source_search_plan and state.multi_source_search_plan.should_search)
    if is_remote_research and _downloaded_pdf_count(state) == 0:
        alerts.append(
            "No PDF/full-text paper was downloaded. Extraction may be based only on metadata, abstracts, README files, or file manifests."
        )

    if is_remote_research and _parseable_downloaded_file_count(state) == 0:
        alerts.append(
            "No parseable downloaded file entered the parser. Consider increasing PDF budget or checking connector failures."
        )

    if _looks_like_mojibake(state.research_question):
        alerts.append(
            "The research question appears to contain mojibake. Use --question-file with a UTF-8 text file for Chinese questions."
        )

    if state.source_triage_decisions and not any(decision.should_ingest for decision in state.source_triage_decisions):
        alerts.append("No discovered source was selected for ingestion; the result is discovery/metadata-only.")

    if state.parsed_sources.figure_assets and not state.chart_extractions:
        alerts.append(
            "Figures were detected in PDFs but no chart data was extracted. "
            "Check Qwen-VL configuration (QWEN_VL_MODEL) or classifier decisions in chart_extractions.json."
        )

    return alerts


def _looks_like_mojibake(text: str) -> bool:
    if not text:
        return False
    suspicious_tokens = ["璇", "鎴", "鍚", "鏂", "€", "锛", "涓", "鐮", "棰"]
    hits = sum(text.count(token) for token in suspicious_tokens)
    return hits >= 3


def _skipped_block_count(processing_log: list[str]) -> int:
    count = 0
    for line in processing_log:
        if "skipped one block" in line:
            count += 1
        match = re.search(r"skipped_blocks=(\d+)", line)
        if match:
            count = max(count, int(match.group(1)))
    return count


def _arxiv_sources_by_filename(state: AgentState) -> dict[str, object]:
    result = {}
    if not state.source_discovery_plan:
        return result
    for source in state.source_discovery_plan.candidate_sources:
        downloaded_path = source.metadata.get("downloaded_path")
        if not downloaded_path:
            continue
        result[Path(str(downloaded_path)).name] = source
    return result


def _join_unique(values) -> str | None:
    cleaned = []
    for value in values:
        if value in (None, "", []):
            continue
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return "; ".join(cleaned) if cleaned else None


def _first_non_empty(*values):
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "table"


def _plain(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _md_cell(value) -> str:
    text = _plain(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text[:240] + "..." if len(text) > 240 else text
