export type TaskStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

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

export interface SourceArtifact {
  artifact_id: string;
  artifact_type?: string;
  url?: string | null;
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
  status: "completed" | "failed";
  research_question: string;
  summary: AgentSummary;
  dynamic_extraction_plan?: DynamicExtractionPlan | null;
  records?: ScientificRecord[];
  dynamic_records?: DynamicRecord[];
  dynamic_records_raw?: DynamicRecord[];
  needs_review_records?: DynamicRecord[];
  source_catalog?: SourceCatalogEntry[];
  connector_status?: Array<Record<string, unknown>>;
  figures?: FigureAsset[];
  chart_extractions?: ChartExtraction[];
  chart_validations?: ChartValidation[];
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
  result?: AgentResult | null;
  summary?: AgentSummary | null;
  quality_report?: QualityReport | null;
  download_urls: Record<string, string>;
  review_decisions: Record<string, ReviewDecision>;
}

export interface ReviewDecision {
  record_id: string;
  decision: "approved" | "needs_changes" | "rejected";
  note?: string | null;
  updated_at: string;
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
  data?: Record<string, unknown>;
}

export interface TaskEventsResponse {
  task_id: string;
  status: TaskStatus;
  events: TaskEvent[];
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
