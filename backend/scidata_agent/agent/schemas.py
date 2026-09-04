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
    relevance_score: float | None = None
    field_scores: dict[str, float] = Field(default_factory=dict)
    relevance_reason: str | None = None
    evidence_types: list[str] = Field(default_factory=list)
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
        "inspected",
        "downloaded",
        "parsed",
        "failed",
        "skipped",
    ] = "discovered"
    parser: str | None = None
    # Parsing is multi-modal: reading PDF text must not make table/figure
    # extraction look complete.  Keep the coarse status for API compatibility,
    # and record each successful operation independently.
    completed_operations: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relevance_score")
    @classmethod
    def clamp_artifact_relevance(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return max(0.0, min(4.0, float(value)))


class ArtifactRelevanceAssessment(BaseModel):
    """LLM assessment of an artifact against the dynamic research needs."""

    artifact_id: str
    topic_alignment: float = 0.0
    task_fit: float = 0.0
    evidence_directness: float = 0.0
    evidence_depth: float = 0.0
    source_authority: float = 0.0
    complementarity: float = 0.0
    overall_score: float = 0.0
    field_scores: dict[str, float] = Field(default_factory=dict)
    evidence_types: list[str] = Field(default_factory=list)
    rank: int | None = None
    recommendation: Literal["process", "inspect_metadata", "skip", "unknown"] = "unknown"
    rationale: str = ""

    @field_validator(
        "topic_alignment",
        "task_fit",
        "evidence_directness",
        "evidence_depth",
        "source_authority",
        "complementarity",
        "overall_score",
    )
    @classmethod
    def clamp_assessment_score(cls, value: float) -> float:
        return max(0.0, min(4.0, float(value)))

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, value: Any) -> Any:
        """Accept concrete planner actions while retaining a strict category."""
        if not isinstance(value, str):
            return value
        canonical = value.strip().casefold().replace("-", "_").replace(" ", "_")
        if canonical in {
            "process",
            "process_artifact",
            "download",
            "download_artifact",
            "parse",
            "parse_pdf",
            "parse_pdf_text",
            "parse_pdf_sections",
            "parse_table",
            "parse_figure",
            "parse_html",
            "parse_csv",
            "read_readme",
            "read_file_manifest",
            "extract_figures",
            "extract_records",
            "extract_dynamic_records",
        }:
            return "process"
        if canonical in {"inspect", "inspect_metadata", "metadata", "read_metadata", "read_source_metadata"}:
            return "inspect_metadata"
        if canonical in {"skip", "ignore", "irrelevant", "reject", "do_not_process"}:
            return "skip"
        if canonical in {"unknown", "unclear", "undecided", "not_sure", "n_a", "na"}:
            return "unknown"
        return value


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
        "inspected",
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
        "plan_task",
        "plan_dynamic_schema",
        "discover_sources",
        "plan_multi_source_search",
        "search_sources",
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
    ]
    purpose: str
    expected_fields: list[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"] = "medium"
    reason: str
    gap_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ArtifactActionPlan(BaseModel):
    """The LLM's next-step plan over the current artifact catalog."""

    research_goal: str
    iteration: int = 0
    should_continue: bool = True
    stop_reason: str | None = None
    actions: list[ArtifactAction] = Field(default_factory=list)
    artifact_assessments: list[ArtifactRelevanceAssessment] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ArtifactActionResult(BaseModel):
    """Auditable outcome of one artifact action execution."""

    action_id: str
    artifact_id: str | None = None
    action: str
    status: Literal["completed", "partial", "skipped", "failed", "no_op"]
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
    # Stable links from retrieval work back to the LLM-authored dynamic table
    # that owns the requested fields.  These fields let the runtime guarantee
    # one initial search and a bounded number of supplemental searches per
    # field group without reinterpreting query text on every turn.
    field_group_id: str | None = None
    target_fields: list[str] = Field(default_factory=list)

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
        "image",
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


class ChartCorrectionResult(BaseModel):
    """Auditable comparison between the initial and second-pass VL reads."""

    figure_id: str
    first_extraction: ChartExtraction
    first_validation: ChartValidationResult
    second_extraction: ChartExtraction | None = None
    second_validation: ChartValidationResult | None = None
    selected_pass: Literal["first", "second"] = "first"
    decision: Literal[
        "accepted_second",
        "kept_first",
        "manual_review",
        "second_pass_failed",
    ] = "kept_first"
    decision_reason: list[str] = Field(default_factory=list)
    needs_review: bool = True


class CrossModalCheck(BaseModel):
    """Auditable comparison between text, table, and figure evidence."""

    check_id: str
    source_file: str
    page: int | None = None
    subject_id: str
    modalities: list[str] = Field(default_factory=list)
    status: Literal["supported", "partial", "not_comparable"] = "not_comparable"
    matched_value_count: int = 0
    candidate_value_count: int = 0
    evidence_refs: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def clamp_cross_modal_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class ReviewQueueItem(BaseModel):
    """A human-review task derived from auditable pipeline risks.

    This is deliberately broader than a record. A reviewer may need to
    inspect a chart validation, a cross-modal mismatch, a source conflict, or
    an evidence-coverage gap. The original extracted values remain elsewhere
    in the result and are never replaced by this item.
    """

    review_id: str
    subject_type: Literal["record", "figure", "cross_modal", "conflict", "coverage_gap"]
    subject_id: str
    priority: Literal["high", "medium", "low"] = "medium"
    risk_type: str
    title: str
    reason: str
    source_file: str | None = None
    page: int | None = None
    record_id: str | None = None
    figure_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ParsedSources(BaseModel):
    text_blocks: list[TextBlock] = Field(default_factory=list)
    heading_candidates: list[HeadingCandidate] = Field(default_factory=list)
    section_plan: SectionPlan | None = None
    section_blocks: list[SectionBlock] = Field(default_factory=list)
    # Stable per-source input fingerprints make section interpretation
    # incremental and allow a changed source to be safely reprocessed.
    section_source_fingerprints: dict[str, str] = Field(default_factory=dict)
    # Figure extraction follows the same per-PDF incremental contract so a
    # later source does not make older figures count against the cap again.
    figure_source_fingerprints: dict[str, str] = Field(default_factory=dict)
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


class EvidenceTrace(BaseModel):
    """A user-facing link from one extracted record to its source evidence."""

    evidence_id: str
    record_id: str
    source_id: str | None = None
    artifact_id: str | None = None
    source_title: str | None = None
    source_file: str
    source_type: str = "unknown"
    page: int | None = None
    section_id: str | None = None
    section_title: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    evidence_type: Literal["text", "table", "figure", "unknown"] = "unknown"
    extraction_method: str | None = None
    evidence_text: str | None = None
    locator_status: Literal["resolved", "partial", "unresolved"] = "unresolved"
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def clamp_evidence_confidence(cls, value: float) -> float:
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
    alignment_context: dict[str, str] = Field(default_factory=dict)
    comparison_basis: list[str] = Field(default_factory=list)
    resolution: Literal["preserve_all", "not_comparable"] = "preserve_all"


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


class CoverageItem(BaseModel):
    name: str
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["covered", "partial", "missing", "unavailable"] = "missing"
    evidence_count: int = 0
    evidence_types: list[str] = Field(default_factory=list)
    reason: str | None = None
    coverage_score: float = 0.0

    @field_validator("coverage_score")
    @classmethod
    def clamp_item_coverage_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class FieldGroupCoverage(BaseModel):
    """Coverage and bounded retrieval status for one dynamic-table field group."""

    group_id: str
    label: str
    fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    coverage_score: float = 0.0
    evidence_count: int = 0
    source_count: int = 0
    initial_search_completed: bool = False
    search_more_count: int = 0
    search_more_limit: int = 2
    status: Literal["pending", "sufficient", "insufficient", "exhausted"] = "pending"

    @field_validator("coverage_score")
    @classmethod
    def clamp_group_coverage_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class CoverageGap(BaseModel):
    """A structured, auditable reason why the workflow should continue."""

    gap_id: str
    requirement_name: str
    priority: Literal["high", "medium", "low"] = "medium"
    status: Literal["missing", "partial", "unavailable"] = "missing"
    missing_fields: list[str] = Field(default_factory=list)
    missing_evidence_types: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    reason: str
    recommended_actions: list[str] = Field(default_factory=list)


class CoverageReport(BaseModel):
    """Deterministic audit of whether the planner may stop."""

    decision: Literal["continue", "allow_stop"] = "continue"
    coverage_score: float = 0.0
    requirements: list[CoverageItem] = Field(default_factory=list)
    gaps: list[CoverageGap] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    required_evidence_types: list[str] = Field(default_factory=list)
    covered_evidence_types: list[str] = Field(default_factory=list)
    unprocessed_relevant_artifacts: list[str] = Field(default_factory=list)
    field_groups: list[FieldGroupCoverage] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("coverage_score")
    @classmethod
    def clamp_coverage_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class ExportFiles(BaseModel):
    csv: str | None = None
    json_file: str | None = Field(default=None, serialization_alias="json")
    processing_log: str | None = None
    quality_report: str | None = None
    coverage_report: str | None = None
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
    evidence_traces_json: str | None = None
    evidence_traces_csv: str | None = None
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
    review_queue_json: str | None = None
    dynamic_tables_dir: str | None = None
    figures_dir: str | None = None
    chart_extractions_json: str | None = None
    chart_validation_json: str | None = None
    chart_corrections_json: str | None = None
    cross_modal_validation_json: str | None = None
    chart_tables_dir: str | None = None
    final_report: str | None = None
    summary_json: str | None = None
    agent_trace_json: str | None = None
    decision_history_json: str | None = None
    tool_history_json: str | None = None


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
    status: Literal["completed", "partial", "failed"] = "completed"
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
    evidence_traces: list[EvidenceTrace] = Field(default_factory=list)
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
    review_queue: list[ReviewQueueItem] = Field(default_factory=list)
    figures: list[FigureAsset] = Field(default_factory=list)
    chart_extractions: list[ChartExtraction] = Field(default_factory=list)
    chart_validations: list[ChartValidationResult] = Field(default_factory=list)
    chart_corrections: list[ChartCorrectionResult] = Field(default_factory=list)
    cross_modal_checks: list[CrossModalCheck] = Field(default_factory=list)
    field_schema: list[dict[str, str]]
    sources: list[SourceSummary]
    processing_log: list[str]
    quality_report: QualityReport
    coverage_report: CoverageReport = Field(default_factory=CoverageReport)
    runtime_iteration: int = 0
    runtime_iteration_budget: int | None = None
    runtime_status: str = "legacy_pipeline"
    runtime_phase: str = "planning"
    runtime_stop_reason: str | None = None
    runtime_no_progress_streak: int = 0
    runtime_no_progress_limit: int = 4
    runtime_last_progress_iteration: int | None = None
    runtime_requires_source_discovery: bool = False
    workflow_revision: int = 0
    runtime_search_more_count: int = 0
    runtime_search_more_limit: int = 2
    runtime_group_initial_searches: list[str] = Field(default_factory=list)
    runtime_group_search_more_counts: dict[str, int] = Field(default_factory=dict)
    runtime_auto_download_sources: bool = True
    runtime_stage_fingerprints: dict[str, str] = Field(default_factory=dict)
    agent_decision_history: list[dict[str, Any]] = Field(default_factory=list)
    tool_result_history: list[dict[str, Any]] = Field(default_factory=list)
    stop_rejections: list[str] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
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
    evidence_traces: list[EvidenceTrace] = Field(default_factory=list)
    artifact_action_plan: ArtifactActionPlan | None = None
    artifact_action_results: list[ArtifactActionResult] = Field(default_factory=list)
    artifact_action_history: list[ArtifactActionIteration] = Field(default_factory=list)
    dynamic_extraction_plan: DynamicExtractionPlan | None = None
    parsed_sources: ParsedSources = Field(default_factory=ParsedSources)
    chart_extractions: list[ChartExtraction] = Field(default_factory=list)
    chart_validations: list[ChartValidationResult] = Field(default_factory=list)
    chart_corrections: list[ChartCorrectionResult] = Field(default_factory=list)
    cross_modal_checks: list[CrossModalCheck] = Field(default_factory=list)
    candidate_records: list[ScientificRecord] = Field(default_factory=list)
    final_records: list[ScientificRecord] = Field(default_factory=list)
    dynamic_records: list[DynamicRecord] = Field(default_factory=list)
    clean_dynamic_records: list[DynamicRecord] = Field(default_factory=list)
    needs_review_records: list[DynamicRecord] = Field(default_factory=list)
    review_queue: list[ReviewQueueItem] = Field(default_factory=list)
    sources: list[SourceSummary] = Field(default_factory=list)
    quality_report: QualityReport = Field(default_factory=QualityReport)
    coverage_report: CoverageReport = Field(default_factory=CoverageReport)
    runtime_iteration: int = 0
    runtime_iteration_budget: int | None = None
    runtime_status: str = "legacy_pipeline"
    runtime_phase: str = "planning"
    runtime_stop_reason: str | None = None
    runtime_no_progress_streak: int = 0
    runtime_no_progress_limit: int = 4
    runtime_last_progress_iteration: int | None = None
    runtime_requires_source_discovery: bool = False
    # A successful search_more starts a new evidence revision.  Completed tool
    # calls from older revisions remain auditable but may be executed again for
    # the newly discovered sources.
    workflow_revision: int = 0
    runtime_search_more_count: int = 0
    runtime_search_more_limit: int = 2
    # Initial retrieval and supplemental-search attempts are tracked by the
    # stable dynamic-table group id.  The legacy aggregate counter remains for
    # audit/UI compatibility but no longer limits unrelated groups.
    runtime_group_initial_searches: list[str] = Field(default_factory=list)
    runtime_group_search_more_counts: dict[str, int] = Field(default_factory=dict)
    runtime_auto_download_sources: bool = True
    # Records which evidence batch each derived-data stage has consumed.  A
    # changed batch reopens the whole extraction/normalization/validation chain.
    runtime_stage_fingerprints: dict[str, str] = Field(default_factory=dict)
    agent_decision_history: list[dict[str, Any]] = Field(default_factory=list)
    tool_result_history: list[dict[str, Any]] = Field(default_factory=list)
    stop_rejections: list[str] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    processing_log: list[str] = Field(default_factory=list)
    export_files: ExportFiles = Field(default_factory=ExportFiles)
    monitor_log_path: Path | None = None

    @field_validator("output_dir")
    @classmethod
    def resolve_output_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()
