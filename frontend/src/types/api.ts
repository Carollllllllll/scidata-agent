export type TaskStatus = "queued" | "running" | "completed" | "partial" | "failed" | "cancelled";

export interface HealthResponse {
  status: "ok";
  service: string;
  version: string;
  qwen_configured: boolean;
  model: string;
  vl_model: string;
  text_model_pool: string[];
  vl_model_pool: string[];
  cors_origins: string[];
  agent_loop: string[];
}

export interface TaskProgress {
  current: number | null;
  total: number | null;
}

export interface UploadedFileInfo {
  name: string;
  size: number;
  content_type?: string | null;
  asset_url?: string;
}

export interface DynamicFieldSpec {
  name: string;
  type: string;
  required: boolean;
  evidence_required: boolean;
  description?: string | null;
  examples?: unknown[];
}

export interface DynamicTableSpec {
  table_name: string;
  description?: string | null;
  entity_type: string;
  priority: "high" | "medium" | "low";
  fields: DynamicFieldSpec[];
}

export interface DynamicExtractionPlan {
  research_goal: string;
  domain: string;
  task_type: string;
  user_focus: string[];
  source_requirements: string[];
  dynamic_tables: DynamicTableSpec[];
  quality_rules: string[];
  missing_data_policy: string;
}

export interface DynamicRecord {
  record_id: string;
  table_name: string;
  fields: Record<string, unknown>;
  source_file: string;
  source_type?: string;
  page?: number | null;
  evidence_text?: string | null;
  confidence: number;
  warnings: string[];
  raw?: Record<string, unknown>;
  paper_title?: string | null;
}

export interface ScientificRecord {
  record_id: string;
  paper_title?: string | null;
  material?: string | null;
  method?: string | null;
  metric_name: string;
  metric_value?: number | null;
  unit?: string | null;
  condition?: string | null;
  source_file: string;
  source_type?: string;
  page?: number | null;
  evidence_text?: string | null;
  confidence: number;
  warnings: string[];
  raw?: Record<string, unknown>;
}

export type EvidenceRecord = DynamicRecord | ScientificRecord;

export interface EvidenceTrace {
  evidence_id: string;
  record_id: string;
  source_id?: string | null;
  artifact_id?: string | null;
  source_title?: string | null;
  source_file: string;
  source_type?: string;
  page?: number | null;
  section_id?: string | null;
  section_title?: string | null;
  table_id?: string | null;
  figure_id?: string | null;
  evidence_type: "text" | "table" | "figure" | "unknown";
  extraction_method?: string | null;
  evidence_text?: string | null;
  locator_status: "resolved" | "partial" | "unresolved";
  confidence: number;
  notes: string[];
}

export interface SourceArtifact {
  artifact_id: string;
  name?: string | null;
  artifact_type?: string;
  size_bytes?: number | null;
  relevance_score?: number | null;
  field_scores?: Record<string, number>;
  relevance_reason?: string | null;
  evidence_types?: string[];
  url?: string | null;
  local_path?: string | null;
  asset_url?: string;
  status?: string;
  parser?: string | null;
  failure_reason?: string | null;
  metadata?: Record<string, unknown>;
}

export interface SourceCatalogEntry {
  source_id: string;
  title: string;
  source_type?: string;
  provider?: string | null;
  url?: string | null;
  status?: string;
  relevance_score?: number;
  selection_action?: string | null;
  triage_action?: string | null;
  reason?: string | null;
  failure_reason?: string | null;
  artifacts?: SourceArtifact[];
  metadata?: Record<string, unknown>;
}

export interface FigureAsset {
  figure_id: string;
  source_file: string;
  page: number;
  label?: string | null;
  caption?: string | null;
  image_url?: string;
  detection_method?: string;
}

export interface ChartAxis {
  label?: string | null;
  unit?: string | null;
  scale?: string;
  range_min?: number | null;
  range_max?: number | null;
}

export interface ChartExtraction {
  extraction_id: string;
  figure_id: string;
  source_file: string;
  page?: number | null;
  chart_type?: string;
  contains_data?: boolean;
  title?: string | null;
  x_axis?: ChartAxis;
  y_axis?: ChartAxis;
  series?: Array<{ name?: string | null; points?: number[][] }>;
  approximate?: boolean;
  confidence?: number;
  notes?: string[];
}

export interface ChartValidation {
  figure_id: string;
  passed: boolean;
  needs_review: boolean;
  issues: Array<{
    severity: string;
    code: string;
    message: string;
    suggestion?: string | null;
  }>;
}

export interface ChartCorrection {
  figure_id: string;
  first_extraction: ChartExtraction;
  first_validation: ChartValidation;
  second_extraction?: ChartExtraction | null;
  second_validation?: ChartValidation | null;
  selected_pass: "first" | "second";
  decision: "accepted_second" | "kept_first" | "manual_review" | "second_pass_failed";
  decision_reason: string[];
  needs_review: boolean;
}

export interface QualityIssue {
  record_id?: string | null;
  level: "info" | "warning" | "error";
  message: string;
  field?: string | null;
}

export interface ConflictIssue {
  conflict_id: string;
  entity?: string | null;
  metric_name: string;
  values: string[];
  record_ids: string[];
  sources: string[];
  message: string;
  alignment_context?: Record<string, string>;
  comparison_basis?: string[];
  resolution?: "preserve_all" | "not_comparable";
}

export interface CrossModalCheck {
  check_id: string;
  source_file: string;
  page?: number | null;
  subject_id: string;
  modalities: string[];
  status: "supported" | "partial" | "not_comparable";
  matched_value_count: number;
  candidate_value_count: number;
  evidence_refs: string[];
  issues: string[];
  confidence: number;
}

export interface QualityReport {
  record_count?: number;
  dynamic_record_count?: number;
  total_record_count?: number;
  issue_count?: number;
  warning_count?: number;
  error_count?: number;
  conflict_count?: number;
  evidence_coverage?: number;
  evidence_text_coverage?: number;
  value_evidence_coverage?: number;
  provenance_page_coverage?: number;
  warning_free_rate?: number;
  review_count?: number;
  field_coverage?: Record<string, number>;
  source_count?: number;
  issues?: QualityIssue[];
  conflicts?: ConflictIssue[];
  notes?: string[];
}

export interface CoverageItem {
  name: string;
  priority: "high" | "medium" | "low";
  status: "covered" | "partial" | "missing" | "unavailable";
  coverage_score?: number;
  evidence_count: number;
  evidence_types: string[];
  reason?: string | null;
}

export interface FieldGroupCoverage {
  group_id: string;
  label: string;
  fields: string[];
  required_fields: string[];
  missing_fields: string[];
  coverage_score: number;
  evidence_count: number;
  source_count: number;
  initial_search_completed: boolean;
  search_more_count: number;
  search_more_limit: number;
  status: "pending" | "sufficient" | "insufficient" | "exhausted";
}

export interface CoverageGap {
  gap_id: string;
  requirement_name: string;
  priority: "high" | "medium" | "low";
  status: "missing" | "partial" | "unavailable";
  missing_fields: string[];
  missing_evidence_types: string[];
  evidence_count: number;
  reason: string;
  recommended_actions: string[];
}

export interface CoverageReport {
  decision: "continue" | "allow_stop";
  coverage_score: number;
  requirements: CoverageItem[];
  gaps: CoverageGap[];
  missing_requirements: string[];
  required_evidence_types: string[];
  covered_evidence_types: string[];
  unprocessed_relevant_artifacts: string[];
  field_groups?: FieldGroupCoverage[];
  reasons: string[];
  recommended_actions: string[];
}

export interface ArtifactRelevanceAssessment {
  artifact_id: string;
  topic_alignment: number;
  task_fit: number;
  evidence_directness: number;
  evidence_depth: number;
  source_authority: number;
  complementarity: number;
  overall_score: number;
  field_scores: Record<string, number>;
  evidence_types: string[];
  rank?: number | null;
  recommendation: "process" | "inspect_metadata" | "skip" | "unknown";
  rationale: string;
}

export interface ArtifactAction {
  action_id: string;
  artifact_id?: string | null;
  action: string;
  purpose: string;
  expected_fields: string[];
  priority: "high" | "medium" | "low";
  reason: string;
  gap_ids: string[];
  parameters: Record<string, unknown>;
}

export interface ArtifactActionResult {
  action_id: string;
  artifact_id?: string | null;
  action: string;
  status: "completed" | "skipped" | "failed" | "no_op";
  message: string;
  output_counts: Record<string, number>;
  warnings: string[];
  error?: string | null;
}

export interface ArtifactActionPlan {
  research_goal: string;
  iteration: number;
  should_continue: boolean;
  stop_reason?: string | null;
  actions: ArtifactAction[];
  artifact_assessments: ArtifactRelevanceAssessment[];
  notes: string[];
}

export interface AgentSummary {
  files_processed?: number;
  text_blocks_processed?: number;
  tables_processed?: number;
  records_extracted?: number;
  records_after_cleaning?: number;
  dynamic_records_extracted?: number;
  dynamic_tables_count?: number;
  figures_detected?: number;
  charts_extracted?: number;
  charts_needs_review?: number;
  warnings?: number;
  [key: string]: unknown;
}

export interface AgentResult {
  task_id: string;
  status: "completed" | "partial" | "failed";
  research_question: string;
  summary: AgentSummary;
  dynamic_extraction_plan?: DynamicExtractionPlan | null;
  records?: ScientificRecord[];
  dynamic_records?: DynamicRecord[];
  dynamic_records_raw?: DynamicRecord[];
  needs_review_records?: DynamicRecord[];
  review_queue?: ReviewQueueItem[];
  source_catalog?: SourceCatalogEntry[];
  evidence_traces?: EvidenceTrace[];
  artifact_action_plan?: ArtifactActionPlan | null;
  artifact_action_results?: ArtifactActionResult[];
  artifact_action_history?: Array<{
    iteration: number;
    plan: ArtifactActionPlan;
    results: ArtifactActionResult[];
  }>;
  coverage_report?: CoverageReport | null;
  connector_status?: Array<Record<string, unknown>>;
  figures?: FigureAsset[];
  chart_extractions?: ChartExtraction[];
  chart_validations?: ChartValidation[];
  chart_corrections?: ChartCorrection[];
  cross_modal_checks?: CrossModalCheck[];
  runtime_iteration?: number;
  runtime_iteration_budget?: number | null;
  runtime_status?: string;
  runtime_phase?: string | null;
  runtime_stop_reason?: string | null;
  runtime_no_progress_streak?: number;
  runtime_no_progress_limit?: number;
  runtime_last_progress_iteration?: number | null;
  runtime_requires_source_discovery?: boolean;
  runtime_search_more_count?: number;
  runtime_search_more_limit?: number;
  runtime_group_initial_searches?: string[];
  runtime_group_search_more_counts?: Record<string, number>;
  agent_decision_history?: AgentDecision[];
  tool_result_history?: ToolResult[];
  stop_rejections?: string[];
  agent_trace?: AgentTraceEvent[];
  quality_report: QualityReport;
  processing_log?: string[];
  export_files?: Record<string, string>;
  download_urls?: Record<string, string>;
  [key: string]: unknown;
}

export interface TaskResponse {
  task_id: string;
  status: TaskStatus;
  research_question?: string | null;
  current_step?: string | null;
  message?: string | null;
  progress?: TaskProgress | null;
  created_at?: string | null;
  updated_at?: string | null;
  error?: { code?: string; message?: string; [key: string]: unknown } | null;
  uploads: UploadedFileInfo[];
  event?: Record<string, unknown> | null;
  runtime?: AgentRuntimeSnapshot | null;
  coverage?: LiveCoverageSnapshot | null;
  source_status?: SourceStatusSnapshot | null;
  result?: AgentResult | null;
  summary?: AgentSummary | null;
  quality_report?: QualityReport | null;
  coverage_report?: CoverageReport | null;
  download_urls: Record<string, string>;
  review_decisions: Record<string, ReviewDecision>;
}

export interface ReviewDecision {
  record_id: string;
  review_id?: string | null;
  subject_id?: string | null;
  subject_type?: string | null;
  decision: "approved" | "needs_changes" | "rejected";
  note?: string | null;
  updated_at: string;
}

export interface AgentDecision {
  decision: "continue" | "stop";
  reason?: string;
  tool_calls?: Array<{
    call_id: string;
    tool_name: string;
    arguments?: Record<string, unknown>;
    reason?: string;
    purpose?: string;
    priority?: "high" | "medium" | "low";
    gap_ids?: string[];
    expected_evidence?: string[];
  }>;
  expected_evidence?: string[];
  stop_reason?: string | null;
}

export interface ToolResult {
  call_id: string;
  tool_name: string;
  status: "completed" | "partial" | "failed" | "skipped";
  data?: Record<string, unknown>;
  artifact_refs?: string[];
  evidence_refs?: string[];
  warnings?: string[];
  errors?: string[];
  elapsed_ms?: number;
  retry_count?: number;
  cached?: boolean;
}

export interface AgentTraceEvent {
  event_id?: string;
  event_type: string;
  iteration?: number;
  call_id?: string | null;
  tool_name?: string | null;
  status?: string | null;
  payload?: Record<string, unknown>;
  created_at?: string;
}

export interface AgentRuntimeSnapshot {
  iteration?: number;
  iteration_budget?: number | null;
  status?: string | null;
  phase?: string | null;
  stop_reason?: string | null;
  no_progress_streak?: number;
  no_progress_limit?: number;
  search_more_count?: number;
  search_more_limit?: number;
  group_initial_searches?: string[];
  group_search_more_counts?: Record<string, number>;
  last_progress_iteration?: number | null;
  decision_count?: number;
  tool_result_count?: number;
  trace_count?: number;
  recent_decisions?: AgentDecision[];
  recent_tool_results?: ToolResult[];
  stop_rejections?: string[];
  latest_event?: {
    event_type?: string;
    iteration?: number;
    call_id?: string | null;
    tool_name?: string | null;
    status?: string | null;
    retry_count?: number;
    cached?: boolean;
    evidence_count?: number;
    artifact_count?: number;
    warning_count?: number;
    error_count?: number;
    reason_count?: number;
  };
}

export interface LiveCoverageSnapshot {
  decision?: "continue" | "allow_stop" | string;
  coverage_score?: number;
  gap_count?: number;
  missing_requirements?: string[];
  required_evidence_types?: string[];
  covered_evidence_types?: string[];
  unprocessed_relevant_artifacts_count?: number;
  field_groups?: FieldGroupCoverage[];
  reasons?: string[];
  recommended_actions?: string[];
}

export interface SourceStatusSnapshot {
  catalog_count?: number | null;
  artifact_count?: number | null;
  source_status_counts?: Record<string, number>;
  artifact_status_counts?: Record<string, number>;
  connectors?: Array<{
    connector?: string;
    connector_name?: string;
    query?: string;
    status?: string;
    attempt?: number;
    attempts?: number;
    retry_count?: number;
    added_sources_count?: number;
    error?: string;
    message?: string;
  }>;
}

export interface ReviewQueueItem {
  review_id: string;
  subject_type: "record" | "figure" | "cross_modal" | "conflict" | "coverage_gap";
  subject_id: string;
  priority: "high" | "medium" | "low";
  risk_type: string;
  title: string;
  reason: string;
  source_file?: string | null;
  page?: number | null;
  record_id?: string | null;
  figure_id?: string | null;
  evidence_refs: string[];
  details: Record<string, unknown>;
}

export interface TaskListResponse {
  tasks: TaskResponse[];
  count: number;
}

export interface TaskEvent {
  timestamp?: string;
  event_type?: string;
  step?: string;
  status?: string;
  message?: string;
  duration_ms?: number;
  data?: {
    runtime?: AgentRuntimeSnapshot;
    coverage?: LiveCoverageSnapshot;
    source_status?: SourceStatusSnapshot;
    [key: string]: unknown;
  };
}

export interface TaskEventsResponse {
  task_id: string;
  status: TaskStatus;
  events: TaskEvent[];
  runtime?: AgentRuntimeSnapshot | null;
  coverage?: LiveCoverageSnapshot | null;
  source_status?: SourceStatusSnapshot | null;
}

export interface TaskSubmissionResponse {
  task_id: string;
  status: TaskStatus;
  status_url: string;
  events_url: string;
}

export interface AnalyzeOptions {
  researchQuestion: string;
  files: File[];
  maxPdfPages: number;
  maxArxivPapers: number | null;
  maxAutoResources: number;
  enableLiveSearch: boolean;
  autoDownloadSources: boolean;
  maxDynamicTextBlocks: number;
  maxRecordTextBlocks: number;
  maxFiguresPerPdf: number;
  reuseDynamicRecordsForMetrics: boolean;
}
