from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scidata_agent.agent.schemas import AgentState, DynamicRecord, EvidenceTrace, ScientificRecord


def build_evidence_traces(state: AgentState) -> list[EvidenceTrace]:
    """Derive auditable record-to-source links without inventing locations.

    The extractor may not know a table, figure, or section for every record. In
    that case the trace remains visible with ``locator_status=unresolved`` so
    the UI can distinguish missing provenance from a successful link.
    """

    records: list[DynamicRecord | ScientificRecord] = []
    seen: set[str] = set()
    for record in [*state.final_records, *(state.clean_dynamic_records or state.dynamic_records)]:
        if record.record_id not in seen:
            records.append(record)
            seen.add(record.record_id)

    traces = [_trace_for_record(state, record) for record in records]
    state.evidence_traces = traces
    return traces


def evidence_trace_rows(traces: list[EvidenceTrace]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": trace.evidence_id,
            "record_id": trace.record_id,
            "source_id": trace.source_id,
            "artifact_id": trace.artifact_id,
            "source_title": trace.source_title,
            "source_file": trace.source_file,
            "source_type": trace.source_type,
            "page": trace.page,
            "section_id": trace.section_id,
            "section_title": trace.section_title,
            "table_id": trace.table_id,
            "figure_id": trace.figure_id,
            "evidence_type": trace.evidence_type,
            "extraction_method": trace.extraction_method,
            "evidence_text": trace.evidence_text,
            "locator_status": trace.locator_status,
            "confidence": trace.confidence,
            "notes": " | ".join(trace.notes),
        }
        for trace in traces
    ]


def _trace_for_record(state: AgentState, record: DynamicRecord | ScientificRecord) -> EvidenceTrace:
    source_file = record.source_file
    page = record.page
    source = _find_source(state, source_file, getattr(record, "paper_title", None))
    catalog_entry = _find_catalog_entry(state, source_file, getattr(record, "paper_title", None))
    artifact = _find_artifact(catalog_entry, source_file)
    section = _find_section(state, source_file, page, record.evidence_text)
    table = _find_table(state, source_file, page)
    figure = _find_figure(state, source_file, page)
    chart = _find_chart(state, figure.figure_id if figure else None, source_file, page)

    evidence_type = "text"
    extraction_method = "text_extraction"
    table_id = None
    figure_id = None
    notes: list[str] = []
    if table is not None:
        evidence_type = "table"
        extraction_method = table.extraction_method
        table_id = table.table_id
    elif figure is not None and chart is not None:
        evidence_type = "figure"
        extraction_method = chart.extraction_method
        figure_id = figure.figure_id
    elif figure is not None:
        notes.append("A figure exists at the same source/page but no chart extraction was linked.")

    if source is None and catalog_entry is None:
        notes.append("No discovered source matched this record.")
    if artifact is None:
        notes.append("No source artifact matched this record.")
    if section is None and page is not None:
        notes.append("Page was retained, but a semantic section was not resolved.")
    if not record.evidence_text:
        notes.append("The record has no evidence text.")

    resolved_parts = sum(
        value is not None
        for value in (source, artifact, page, section, table_id, figure_id, record.evidence_text)
    )
    locator_status = "resolved" if resolved_parts >= 4 else "partial" if resolved_parts > 1 else "unresolved"
    return EvidenceTrace(
        evidence_id=_stable_id(record.record_id, source_file, page, table_id, figure_id),
        record_id=record.record_id,
        source_id=source.source_id if source else catalog_entry.source_id if catalog_entry else None,
        artifact_id=artifact.artifact_id if artifact else None,
        source_title=source.title if source else catalog_entry.title if catalog_entry else getattr(record, "paper_title", None),
        source_file=source_file,
        source_type=record.source_type.value if hasattr(record.source_type, "value") else str(record.source_type),
        page=page,
        section_id=section.section_id if section else None,
        section_title=section.section_title if section else None,
        table_id=table_id,
        figure_id=figure_id,
        evidence_type=evidence_type if record.evidence_text or table or figure else "unknown",
        extraction_method=extraction_method if record.evidence_text or table or chart else None,
        evidence_text=record.evidence_text,
        locator_status=locator_status,
        confidence=record.confidence,
        notes=notes,
    )


def _find_source(state: AgentState, source_file: str, paper_title: str | None):
    filename = Path(source_file).name.casefold()
    title = (paper_title or "").casefold().strip()
    sources = state.source_discovery_plan.candidate_sources if state.source_discovery_plan else []
    for source in sources:
        if title and source.title.casefold() == title:
            return source
        metadata = source.metadata or {}
        paths = [metadata.get("downloaded_path"), *(metadata.get("downloaded_paths") or [])]
        if any(Path(str(path)).name.casefold() == filename for path in paths if path):
            return source
        if filename and filename in source.title.casefold():
            return source
    return None


def _find_artifact(source, source_file: str):
    if source is None:
        return None
    filename = Path(source_file).name.casefold()
    for artifact in getattr(source, "artifacts", []) or []:
        artifact_name = Path(str(artifact.name or artifact.local_path or "")).name.casefold()
        if artifact_name == filename:
            return artifact
    return None


def _find_catalog_entry(state: AgentState, source_file: str, paper_title: str | None):
    filename = Path(source_file).name.casefold()
    title = (paper_title or "").casefold().strip()
    for entry in state.source_catalog:
        if title and entry.title.casefold() == title:
            return entry
        for artifact in entry.artifacts:
            if Path(str(artifact.name or artifact.local_path or "")).name.casefold() == filename:
                return entry
        if filename and filename in entry.title.casefold():
            return entry
    return None


def _find_section(state: AgentState, source_file: str, page: int | None, evidence_text: str | None):
    blocks = [block for block in state.parsed_sources.section_blocks if Path(block.source_file).name == Path(source_file).name]
    if evidence_text:
        snippet = " ".join(evidence_text.split())[:120]
        for block in blocks:
            if snippet and snippet in " ".join(block.text.split()):
                return block
    for block in blocks:
        if page is not None and block.page_start is not None and block.page_end is not None and block.page_start <= page <= block.page_end:
            return block
    return blocks[0] if len(blocks) == 1 else None


def _find_table(state: AgentState, source_file: str, page: int | None):
    tables = [table for table in state.parsed_sources.tables if Path(table.source_file).name == Path(source_file).name]
    if page is not None:
        page_tables = [table for table in tables if table.page == page]
        if page_tables:
            return page_tables[0]
    return tables[0] if len(tables) == 1 else None


def _find_figure(state: AgentState, source_file: str, page: int | None):
    figures = [figure for figure in state.parsed_sources.figure_assets if Path(figure.source_file).name == Path(source_file).name]
    if page is not None:
        page_figures = [figure for figure in figures if figure.page == page]
        if page_figures:
            return page_figures[0]
    return figures[0] if len(figures) == 1 else None


def _find_chart(state: AgentState, figure_id: str | None, source_file: str, page: int | None):
    for chart in state.chart_extractions:
        if figure_id and chart.figure_id == figure_id:
            return chart
        if chart.source_file == source_file and chart.page == page:
            return chart
    return None


def _stable_id(*parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"evidence_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"
