from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def timestamp_task_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{timestamp}_{uuid4().hex[:4]}"


class SourceType(str, Enum):
    PDF_TEXT = "pdf_text"
    PDF_TABLE = "pdf_table"
    CSV = "csv"
    EXCEL = "excel"
    FIGURE_CHART = "figure_chart"
    UNKNOWN = "unknown"


class UploadedFile(BaseModel):
    file_id: str = Field(default_factory=lambda: f"file_{uuid4().hex[:8]}")
    filename: str
    path: Path
    content_type: str | None = None

    @field_validator("path")
    @classmethod
    def expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class TaskPlan(BaseModel):
    domain: str = "scientific data extraction"
    research_goal: str | None = None
    target_fields: list[str] = Field(default_factory=list)
    dynamic_schema: dict[str, Any] = Field(default_factory=dict)
    source_requirements: list[str] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    output_format: list[Literal["csv", "json", "kg"]] = Field(default_factory=lambda: ["csv", "json"])
    need_provenance: bool = True
    assumptions: list[str] = Field(default_factory=list)
    schema_notes: list[str] = Field(default_factory=list)


class DiscoveredSource(BaseModel):
    source_id: str = Field(default_factory=lambda: f"src_{uuid4().hex[:8]}")
    # Stable identity for the real-world paper/dataset represented by one or
    # more provider records. ``source_id`` remains the canonical provider row.
    source_cluster_id: str | None = None
    title: str
    source_type: Literal[
        "paper",
        "paper_search",
        "paper_metadata",
        "open_database",
        "dataset",
        "supplementary_material",
        "table",
        "image",
        "webpage",
        "repository",
        "unknown",
    ] = "unknown"
    url: str | None = None
    query: str | None = None
    description: str | None = None
    reason: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_source_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class SourceArtifact(BaseModel):
    """A concrete readable material belonging to a discovered source."""

    artifact_id: str = Field(default_factory=lambda: f"artifact_{uuid4().hex[:10]}")
    source_id: str
    source_cluster_id: str | None = None
    provider: str | None = None
    name: str | None = None
    size_bytes: int | None = None
    artifact_type: Literal[
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
    ] = "unknown"
    url: str | None = None
    local_path: str | None = None
    content_type: str | None = None
    status: Literal[
        "discovered",
        "selected",
        "planned",
        "metadata_read",
        "downloaded",
        "parsed",
        "failed",
        "skipped",
    ] = "discovered"
    parser: str | None = None
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceCatalogEntry(BaseModel):
    """Normalized source plus its artifacts and current execution state."""

    source_id: str
    source_cluster_id: str | None = None
    title: str
    source_type: str = "unknown"
    provider: str | None = None
    url: str | None = None
    status: Literal[
        "discovered",
        "selected",
        "metadata_read",
        "downloaded",
        "parsed",
        "failed",
        "skipped",
    ] = "discovered"
    relevance_score: float = 0.0
    selection_action: str | None = None
    triage_action: str | None = None
    reason: str | None = None
    failure_reason: str | None = None
    artifacts: list[SourceArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relevance_score")
    @classmethod
    def clamp_catalog_relevance(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class ArtifactAction(BaseModel):
    """One model-selected operation against a concrete source artifact."""

    action_id: str
    artifact_id: str | None = None
    action: Literal[
        "read_metadata",
        "download_artifact",
        "parse_pdf_text",
        "parse_pdf_sections",
        "parse_table",
        "parse_figure",
        "parse_html",
        "parse_csv",
        "read_readme",
        "read_file_manifest",
        "search_more",
        "validate_evidence",
        "stop",
    ]
    purpose: str
    expected_fields: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    reason: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ArtifactActionPlan(BaseModel):
    """The LLM's next-step plan over the current artifact catalog."""

    research_goal: str
    iteration: int = 0
    should_continue: bool = True
    stop_reason: str | None = None
    actions: list[ArtifactAction] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArtifactActionResult(BaseModel):
    """Auditable outcome of one artifact action execution."""

    action_id: str
    artifact_id: str | None = None
    action: str
    status: Literal["completed", "skipped", "failed", "no_op"]
    message: str
    output_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ArtifactActionIteration(BaseModel):
    """One auditable planner/executor iteration."""

    iteration: int
    plan: ArtifactActionPlan
    results: list[ArtifactActionResult] = Field(default_factory=list)


class SourceDiscoveryPlan(BaseModel):
    research_goal: str
    domain: str = "general science"
    recommended_keywords: list[str] = Field(default_factory=list)
    target_data_types: list[str] = Field(default_factory=list)
    dynamic_schema: dict[str, Any] = Field(default_factory=dict)
    candidate_sources: list[DiscoveredSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArxivSearchQuery(BaseModel):
    query: str
    purpose: str | None = None
    max_results: int = 100

    @field_validator("query")
    @classmethod
    def require_query(cls, value: str) -> str:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            raise ValueError("arXiv query must not be empty")
        return cleaned

    @field_validator("max_results")
    @classmethod
    def clamp_max_results(cls, value: int) -> int:
        return max(1, int(value))


class ArxivSearchPlan(BaseModel):
    research_goal: str
    should_search_arxiv: bool = True
    search_intent: str | None = None
    queries: list[ArxivSearchQuery] = Field(default_factory=list)
    selection_criteria: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SourceSearchRequest(BaseModel):
    connector_name: Literal[
        "arxiv",
        "openalex",
        "semantic_scholar",
        "crossref",
        "zenodo",
        "figshare",
        "github",
    ]
    source_type: Literal[
        "paper",
        "paper_search",
        "paper_metadata",
        "open_database",
        "dataset",
        "supplementary_material",
        "table",
        "image",
        "webpage",
        "repository",
        "unknown",
    ] = "unknown"
    query: str
    purpose: str | None = None
    max_results: int = 100
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def require_query(cls, value: str) -> str:
        cleaned = " ".join(str(value).split())
        if not cleaned:
            raise ValueError("source search query must not be empty")
        return cleaned

    @field_validator("max_results")
    @classmethod
    def clamp_max_results(cls, value: int) -> int:
        return max(1, int(value))


class MultiSourceSearchPlan(BaseModel):
    research_goal: str
    domain: str = "general science"
    should_search: bool = True
    search_requests: list[SourceSearchRequest] = Field(default_factory=list)
    selection_criteria: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SourceTriageDecision(BaseModel):
    source_id: str
    title: str
    provider: str | None = None
    source_type: str = "unknown"
    relevance_score: float = 0.0
    recommended_action: Literal[
        "record_only",
        "read_metadata",
        "read_readme",
        "read_file_manifest",
        "download_pdf",
        "download_small_table",
        "download_small_supplement",
        "ask_user",
        "skip",
    ] = "record_only"
    reason: str | None = None
    estimated_download_size: int | None = None
    estimated_cost: Literal["low", "medium", "high", "unknown"] = "unknown"
    risk: Literal["low", "medium", "high"] = "low"
    should_ingest: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relevance_score")
    @classmethod
    def clamp_relevance_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class SourceSelectionDecision(BaseModel):
    source_id: str
    decision: Literal[
        "deep_read",
        "metadata_only",
        "read_readme",
        "read_file_manifest",
        "download_small_table",
        "download_small_supplement",
        "reject",
        "ask_user",
    ] = "metadata_only"
    priority: Literal["high", "medium", "low"] = "medium"
    source_role: Literal[
        "primary_paper",
        "supporting_paper",
        "dataset",
        "supplementary_material",
        "code_repository",
        "metadata_reference",
        "noise",
        "unknown",
    ] = "unknown"
    priority_score: float = 0.5
    reason: str
    matched_requirements: list[str] = Field(default_factory=list)
    expected_extractable_fields: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)

    @field_validator("priority_score")
    @classmethod
    def clamp_priority_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class SourceSelectionPlan(BaseModel):
    research_goal: str
    selection_summary: str | None = None
    time_range_interpreted: str | None = None
    decisions: list[SourceSelectionDecision] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SourceInsight(BaseModel):
    insight_id: str = Field(default_factory=lambda: f"insight_{uuid4().hex[:8]}")
    source_id: str
    title: str
    provider: str | None = None
    source_type: str = "unknown"
    insight_type: Literal[
        "metadata",
        "abstract",
        "file_manifest",
        "readme",
        "downloaded_file",
        "download_error",
        "source_summary",
    ] = "metadata"
    content: str
    url: str | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_insight_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class TextBlock(BaseModel):
    source_file: str
    source_path: str
    source_type: SourceType = SourceType.PDF_TEXT
    page: int | None = None
    text: str
    chunk_id: str


class HeadingCandidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: f"heading_{uuid4().hex[:8]}")
    source_file: str
    source_path: str
    page: int
    line_index: int
    text: str
    before_text: str | None = None
    after_text: str | None = None
    font_size: float | None = None
    is_bold: bool = False
    y_position: float | None = None
    extraction_method: Literal["layout", "text"] = "text"
    score: float = 0.0


class SectionPlanItem(BaseModel):
    source_file: str | None = None
    section_title: str
    section_type: str = "other"
    start_page: int
    start_anchor: str
    confidence: float = 0.5
    reason: str | None = None

    @field_validator("confidence")
    @classmethod
    def clamp_section_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class IgnoredHeadingCandidate(BaseModel):
    text: str
    page: int | None = None
    reason: str | None = None


class SectionPlan(BaseModel):
    sections: list[SectionPlanItem] = Field(default_factory=list)
    ignored_candidates: list[IgnoredHeadingCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    used_llm: bool = True


class SectionBlock(BaseModel):
    source_file: str
    source_path: str
    source_type: SourceType = SourceType.PDF_TEXT
    section_id: str = Field(default_factory=lambda: f"section_{uuid4().hex[:8]}")
    section_title: str | None = None
    section_type: str = "other"
    page_start: int | None = None
    page_end: int | None = None
    page: int | None = None
    text: str
    chunk_id: str
    confidence: float = 0.5
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_section_block_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class TableBlock(BaseModel):
    source_file: str
    source_path: str
    source_type: SourceType
    columns: list[str]
    rows: list[dict[str, Any]]
    table_id: str
    page: int | None = None
    caption: str | None = None
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in PDF points
    extraction_method: str = "parser"  # parser | pdfplumber_lines | pdfplumber_text
    raw: dict[str, Any] = Field(default_factory=dict)


class FigureAsset(BaseModel):
    """A figure/chart located inside a PDF and rendered to a local image."""

    figure_id: str = Field(default_factory=lambda: f"fig_{uuid4().hex[:8]}")
    source_file: str
    source_path: str
    page: int
    label: str | None = None  # e.g. "Figure 3"
    caption: str | None = None  # caption text, doubles as provenance evidence
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in PDF points
    image_path: str | None = None  # rendered PNG path
    detection_method: str = "caption"  # caption | graphics | caption+graphics


class ChartAxisSpec(BaseModel):
    label: str | None = None
    unit: str | None = None
    scale: Literal["linear", "log", "unknown"] = "unknown"
    range_min: float | None = None
    range_max: float | None = None


class ChartSeries(BaseModel):
    name: str | None = None
    points: list[list[float]] = Field(default_factory=list)  # [[x, y], ...]
    point_style: str | None = None


class ChartExtraction(BaseModel):
    """Structured chart data extracted from a figure by the vision LLM.

    Values read from chart pixels are approximations; downstream consumers
    must treat ``approximate=True`` values as estimates, not exact data.
    """

    extraction_id: str = Field(default_factory=lambda: f"chart_{uuid4().hex[:8]}")
    figure_id: str
    source_file: str
    page: int | None = None
    chart_type: str = "unknown"  # line | scatter | bar | heatmap | contour | image | other
    contains_data: bool = False
    title: str | None = None
    x_axis: ChartAxisSpec = Field(default_factory=ChartAxisSpec)
    y_axis: ChartAxisSpec = Field(default_factory=ChartAxisSpec)
    series: list[ChartSeries] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    approximate: bool = True
    confidence: float = 0.0
    extraction_method: str = "qwen_vl"
    raw_response_excerpt: str | None = None


class ChartValidationIssue(BaseModel):
    severity: Literal["info", "warning", "error"] = "warning"
    code: str  # axis_range_mismatch | series_count_mismatch | unit_suspect | low_confidence | ...
    message: str
    suggestion: str | None = None


class ChartValidationResult(BaseModel):
    figure_id: str
    passed: bool = True
    issues: list[ChartValidationIssue] = Field(default_factory=list)
    needs_review: bool = False
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ParsedSources(BaseModel):
    text_blocks: list[TextBlock] = Field(default_factory=list)
    heading_candidates: list[HeadingCandidate] = Field(default_factory=list)
    section_plan: SectionPlan | None = None
    section_blocks: list[SectionBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    figure_assets: list[FigureAsset] = Field(default_factory=list)
    parser_warnings: list[str] = Field(default_factory=list)
    table_extraction_status: list[dict[str, Any]] = Field(default_factory=list)
    # Map source_file -> extracted full paper title. Used to backfill records
    # that do not carry an explicit paper title.
    file_titles: dict[str, str] = Field(default_factory=dict)


class DynamicFieldSpec(BaseModel):
    name: str
    type: str = "string"
    required: bool = False
    evidence_required: bool = True
    description: str | None = None
    examples: list[Any] = Field(default_factory=list)


class DynamicTableSpec(BaseModel):
    table_name: str
    description: str | None = None
    entity_type: str = "other"
    priority: Literal["high", "medium", "low"] = "medium"
    fields: list[DynamicFieldSpec] = Field(default_factory=list)


class InformationNeed(BaseModel):
    need_name: str
    reason: str | None = None
    priority: Literal["high", "medium", "low"] = "medium"


class DynamicExtractionPlan(BaseModel):
    research_goal: str
    domain: str = "general science"
    task_type: str = "literature_survey"
    user_focus: list[str] = Field(default_factory=list)
    time_range: str | None = None
    source_requirements: list[str] = Field(default_factory=list)
    information_needs: list[InformationNeed] = Field(default_factory=list)
    dynamic_tables: list[DynamicTableSpec] = Field(default_factory=list)
    quality_rules: list[str] = Field(default_factory=list)
    missing_data_policy: str = "Use null for missing information; do not fabricate values."


class DynamicRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: f"dyn_{uuid4().hex[:10]}")
    table_name: str
    fields: dict[str, Any] = Field(default_factory=dict)
    source_file: str
    source_type: SourceType = SourceType.UNKNOWN
    page: int | None = None
    evidence_text: str | None = None
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    paper_title: str | None = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class ScientificRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: f"rec_{uuid4().hex[:10]}")
    paper_title: str | None = None
    material: str | None = None
    method: str | None = None
    metric_name: str
    metric_value: float | None = None
    unit: str | None = None
    condition: str | None = None
    source_file: str
    source_type: SourceType = SourceType.UNKNOWN
    page: int | None = None
    evidence_text: str | None = None
    confidence: float = 0.5
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class SourceSummary(BaseModel):
    source_file: str
    source_path: str
    source_type: SourceType
    pages_processed: list[int] = Field(default_factory=list)
    tables_processed: int = 0
    records_count: int = 0
    notes: list[str] = Field(default_factory=list)


class QualityIssue(BaseModel):
    record_id: str | None = None
    level: Literal["info", "warning", "error"]
    message: str
    field: str | None = None


class ConflictIssue(BaseModel):
    conflict_id: str = Field(default_factory=lambda: f"conflict_{uuid4().hex[:8]}")
    entity: str | None = None
    metric_name: str
    values: list[str] = Field(default_factory=list)
    record_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    message: str


class QualityReport(BaseModel):
    record_count: int = 0
    dynamic_record_count: int = 0
    total_record_count: int = 0
    issue_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    conflict_count: int = 0
    evidence_coverage: float = 0.0
    evidence_text_coverage: float = 0.0
    value_evidence_coverage: float = 0.0
    provenance_page_coverage: float = 0.0
    warning_free_rate: float = 0.0
    review_count: int = 0
    field_coverage: dict[str, float] = Field(default_factory=dict)
    source_count: int = 0
    issues: list[QualityIssue] = Field(default_factory=list)
    conflicts: list[ConflictIssue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ExportFiles(BaseModel):
    csv: str | None = None
    json_file: str | None = Field(default=None, serialization_alias="json")
    processing_log: str | None = None
    quality_report: str | None = None
    source_discovery_plan: str | None = None
    arxiv_search_plan: str | None = None
    multi_source_search_plan: str | None = None
    connector_status_csv: str | None = None
    connector_status_json: str | None = None
    source_selection_plan: str | None = None
    source_selection_csv: str | None = None
    discovered_sources_csv: str | None = None
    discovered_sources_json: str | None = None
    source_triage_csv: str | None = None
    source_triage_json: str | None = None
    source_research_csv: str | None = None
    source_research_json: str | None = None
    source_catalog_json: str | None = None
    source_catalog_csv: str | None = None
    artifact_action_plan_json: str | None = None
    artifact_action_results_json: str | None = None
    artifact_action_history_json: str | None = None
    section_plan: str | None = None
    monitor_log: str | None = None
    paper_survey_csv: str | None = None
    paper_survey_json: str | None = None
    dynamic_schema: str | None = None
    dynamic_records: str | None = None
    clean_dynamic_records: str | None = None
    needs_review: str | None = None
    dynamic_tables_dir: str | None = None
    figures_dir: str | None = None
    chart_extractions_json: str | None = None
    chart_validation_json: str | None = None
    chart_tables_dir: str | None = None
    final_report: str | None = None
    summary_json: str | None = None


class AgentSummary(BaseModel):
    files_processed: int = 0
    text_blocks_processed: int = 0
    heading_candidates_extracted: int = 0
    section_blocks_processed: int = 0
    tables_processed: int = 0
    records_extracted: int = 0
    records_after_cleaning: int = 0
    dynamic_records_extracted: int = 0
    dynamic_tables_count: int = 0
    figures_detected: int = 0
    charts_extracted: int = 0
    charts_needs_review: int = 0
    warnings: int = 0


class AgentResult(BaseModel):
    task_id: str
    status: Literal["completed", "failed"] = "completed"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    research_question: str
    task_plan: TaskPlan
    source_discovery_plan: SourceDiscoveryPlan | None = None
    arxiv_search_plan: ArxivSearchPlan | None = None
    multi_source_search_plan: MultiSourceSearchPlan | None = None
    source_selection_plan: SourceSelectionPlan | None = None
    source_triage_decisions: list[SourceTriageDecision] = Field(default_factory=list)
    source_insights: list[SourceInsight] = Field(default_factory=list)
    source_catalog: list[SourceCatalogEntry] = Field(default_factory=list)
    artifact_action_plan: ArtifactActionPlan | None = None
    artifact_action_results: list[ArtifactActionResult] = Field(default_factory=list)
    artifact_action_history: list[ArtifactActionIteration] = Field(default_factory=list)
    dynamic_extraction_plan: DynamicExtractionPlan | None = None
    connector_status: list[dict[str, Any]] = Field(default_factory=list)
    summary: AgentSummary
    records: list[ScientificRecord]
    dynamic_records: list[DynamicRecord] = Field(default_factory=list)
    dynamic_records_raw: list[DynamicRecord] = Field(default_factory=list)
    needs_review_records: list[DynamicRecord] = Field(default_factory=list)
    figures: list[FigureAsset] = Field(default_factory=list)
    chart_extractions: list[ChartExtraction] = Field(default_factory=list)
    chart_validations: list[ChartValidationResult] = Field(default_factory=list)
    field_schema: list[dict[str, str]]
    sources: list[SourceSummary]
    processing_log: list[str]
    quality_report: QualityReport
    export_files: ExportFiles


class AgentState(BaseModel):
    task_id: str = Field(default_factory=timestamp_task_id)
    research_question: str
    files: list[UploadedFile]
    output_dir: Path
    task_plan: TaskPlan | None = None
    source_discovery_plan: SourceDiscoveryPlan | None = None
    arxiv_search_plan: ArxivSearchPlan | None = None
    multi_source_search_plan: MultiSourceSearchPlan | None = None
    connector_status: list[dict[str, Any]] = Field(default_factory=list)
    source_selection_plan: SourceSelectionPlan | None = None
    source_triage_decisions: list[SourceTriageDecision] = Field(default_factory=list)
    source_insights: list[SourceInsight] = Field(default_factory=list)
    source_catalog: list[SourceCatalogEntry] = Field(default_factory=list)
    artifact_action_plan: ArtifactActionPlan | None = None
    artifact_action_results: list[ArtifactActionResult] = Field(default_factory=list)
    artifact_action_history: list[ArtifactActionIteration] = Field(default_factory=list)
    dynamic_extraction_plan: DynamicExtractionPlan | None = None
    parsed_sources: ParsedSources = Field(default_factory=ParsedSources)
    chart_extractions: list[ChartExtraction] = Field(default_factory=list)
    chart_validations: list[ChartValidationResult] = Field(default_factory=list)
    candidate_records: list[ScientificRecord] = Field(default_factory=list)
    final_records: list[ScientificRecord] = Field(default_factory=list)
    dynamic_records: list[DynamicRecord] = Field(default_factory=list)
    clean_dynamic_records: list[DynamicRecord] = Field(default_factory=list)
    needs_review_records: list[DynamicRecord] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    quality_report: QualityReport = Field(default_factory=QualityReport)
    processing_log: list[str] = Field(default_factory=list)
    export_files: ExportFiles = Field(default_factory=ExportFiles)
    monitor_log_path: Path | None = None

    @field_validator("output_dir")
    @classmethod
    def resolve_output_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()
