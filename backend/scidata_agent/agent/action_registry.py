from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCapability:
    """Static capability metadata used to validate artifact routing."""

    action: str
    description: str
    artifact_types: frozenset[str] = frozenset()
    requires_local_path: bool = False
    global_action: bool = False


_GLOBAL_ARTIFACT_ACTIONS = frozenset({
    "plan_task",
    "plan_dynamic_schema",
    "discover_sources",
    "plan_multi_source_search",
    "search_sources",
    "search_more",
    "validate_evidence",
    "select_sources",
    "triage_sources",
    "ingest_sources",
    "ingest_arxiv_pdfs",
    "parse_content",
    "parse_source_content",
    "extract_figures",
    "interpret_sections",
    "extract_dynamic_records",
    "extract_records",
    "normalize_records",
    "track_provenance",
    "validate_quality",
    "stop",
})

ACTION_CAPABILITIES: dict[str, ActionCapability] = {
    "plan_task": ActionCapability(
        action="plan_task",
        description="Translate the research question into a task-specific scientific extraction plan.",
        global_action=True,
    ),
    "plan_dynamic_schema": ActionCapability(
        action="plan_dynamic_schema",
        description="Design the dynamic tables, fields, evidence requirements, and quality rules for the task.",
        global_action=True,
    ),
    "discover_sources": ActionCapability(
        action="discover_sources",
        description="Discover candidate papers, datasets, repositories, and other evidence sources for the task.",
        global_action=True,
    ),
    "plan_multi_source_search": ActionCapability(
        action="plan_multi_source_search",
        description="Create connector-specific queries for the discovered source landscape.",
        global_action=True,
    ),
    "search_sources": ActionCapability(
        action="search_sources",
        description="Execute the current multi-source search plan and merge returned source metadata.",
        global_action=True,
    ),
    "read_metadata": ActionCapability(
        action="read_metadata",
        description="Read catalog metadata without downloading the artifact.",
    ),
    "download_artifact": ActionCapability(
        action="download_artifact",
        description="Download one selected remote artifact into the task workspace.",
        artifact_types=frozenset({
            "landing_page",
            "pdf",
            "html",
            "csv",
            "tsv",
            "xlsx",
            "json",
            "xml",
            "readme",
            "code_archive",
            "supplementary_pdf",
            "image",
            "file_manifest",
            "unknown",
        }),
    ),
    "parse_pdf_text": ActionCapability(
        action="parse_pdf_text",
        description="Extract text blocks from a PDF artifact.",
        artifact_types=frozenset({"pdf", "supplementary_pdf"}),
        requires_local_path=True,
    ),
    "parse_pdf_sections": ActionCapability(
        action="parse_pdf_sections",
        description="Extract PDF text and create LLM-interpreted section blocks.",
        artifact_types=frozenset({"pdf", "supplementary_pdf"}),
        requires_local_path=True,
    ),
    "parse_table": ActionCapability(
        action="parse_table",
        description="Extract a PDF table with TATR/pdfplumber or read a spreadsheet table.",
        artifact_types=frozenset({"pdf", "supplementary_pdf", "csv", "tsv", "xlsx", "json", "xml"}),
        requires_local_path=True,
    ),
    "parse_figure": ActionCapability(
        action="parse_figure",
        description="Locate and interpret quantitative figure/chart evidence with VL.",
        artifact_types=frozenset({"pdf", "supplementary_pdf", "image"}),
        requires_local_path=True,
    ),
    "parse_html": ActionCapability(
        action="parse_html",
        description="Read a local HTML artifact as text evidence.",
        artifact_types=frozenset({"html", "landing_page"}),
        requires_local_path=True,
    ),
    "parse_csv": ActionCapability(
        action="parse_csv",
        description="Read a CSV or TSV artifact as a structured table.",
        artifact_types=frozenset({"csv", "tsv"}),
        requires_local_path=True,
    ),
    "read_readme": ActionCapability(
        action="read_readme",
        description="Read a local README or code artifact as text evidence.",
        artifact_types=frozenset({"readme", "code_archive"}),
        requires_local_path=True,
    ),
    "read_file_manifest": ActionCapability(
        action="read_file_manifest",
        description="Inspect the file manifest already attached to a source artifact.",
        artifact_types=frozenset({"file_manifest", "landing_page", "code_archive"}),
    ),
    "search_more": ActionCapability(
        action="search_more",
        description="Request another discovery/search iteration.",
        global_action=True,
    ),
    "validate_evidence": ActionCapability(
        action="validate_evidence",
        description="Run the existing evidence and quality validation node.",
        global_action=True,
    ),
    "select_sources": ActionCapability(
        action="select_sources",
        description="Compare discovered sources and select those that match the research goal.",
        global_action=True,
    ),
    "triage_sources": ActionCapability(
        action="triage_sources",
        description="Classify selected sources and decide which artifacts should be ingested.",
        global_action=True,
    ),
    "ingest_sources": ActionCapability(
        action="ingest_sources",
        description="Materialize selected non-arXiv source artifacts for evidence extraction.",
        global_action=True,
    ),
    "ingest_arxiv_pdfs": ActionCapability(
        action="ingest_arxiv_pdfs",
        description="Download selected arXiv PDFs with bounded timeout and resumable storage.",
        global_action=True,
    ),
    "parse_content": ActionCapability(
        action="parse_content",
        description="Run the existing PDF, table, figure, chart, and dynamic-record extraction pipeline.",
        global_action=True,
    ),
    "parse_source_content": ActionCapability(
        action="parse_source_content",
        description="Parse currently materialized source files into text blocks, headings, and tables.",
        global_action=True,
    ),
    "extract_figures": ActionCapability(
        action="extract_figures",
        description="Locate and interpret quantitative figures and charts with the configured VL model.",
        global_action=True,
    ),
    "interpret_sections": ActionCapability(
        action="interpret_sections",
        description="Interpret parsed text blocks into section-aware evidence blocks.",
        global_action=True,
    ),
    "extract_dynamic_records": ActionCapability(
        action="extract_dynamic_records",
        description="Extract task-specific dynamic records from text, tables, and figure evidence.",
        global_action=True,
    ),
    "extract_records": ActionCapability(
        action="extract_records",
        description="Extract the compatibility scientific record representation from available evidence.",
        global_action=True,
    ),
    "normalize_records": ActionCapability(
        action="normalize_records",
        description="Normalize extracted scientific records while preserving provenance and uncertainty.",
        global_action=True,
    ),
    "track_provenance": ActionCapability(
        action="track_provenance",
        description="Build source summaries and evidence traces for the current records.",
        global_action=True,
    ),
    "validate_quality": ActionCapability(
        action="validate_quality",
        description="Run deterministic quality, conflict, coverage, and evidence validation.",
        global_action=True,
    ),
    "stop": ActionCapability(
        action="stop",
        description="Stop the current planner iteration.",
        global_action=True,
    ),
}


def get_action_capability(action: str) -> ActionCapability:
    try:
        return ACTION_CAPABILITIES[action]
    except KeyError as exc:
        raise ValueError(f"Unsupported artifact action: {action!r}") from exc


def list_action_capabilities() -> list[ActionCapability]:
    return list(ACTION_CAPABILITIES.values())


def is_global_action(action: str) -> bool:
    return action in _GLOBAL_ARTIFACT_ACTIONS


def artifact_type_supported(action: str, artifact_type: str) -> bool:
    capability = get_action_capability(action)
    if capability.global_action or action == "read_metadata":
        return True
    return artifact_type in capability.artifact_types
