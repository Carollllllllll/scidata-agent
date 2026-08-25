import type { TaskEvent, TaskProgress, TaskStatus } from "../types/api";

export const AGENT_STAGES = [
  "ensure_llm_ready",
  "task_planning",
  "dynamic_schema_planning",
  "source_discovery",
  "multi_source_search_planning",
  "multi_source_search",
  "source_selection",
  "source_triage",
  "multi_source_ingestion",
  "arxiv_pdf_ingestion",
  "artifact_action_planning",
  "artifact_action_execution",
  "source_parsing",
  "figure_chart_extraction",
  "section_interpretation",
  "dynamic_extraction",
  "record_extraction",
  "normalization",
  "provenance_tracking",
  "quality_validation",
  "export",
] as const;

export const STAGE_LABELS: Record<string, string> = {
  queued: "等待执行",
  starting: "启动任务",
  ensure_llm_ready: "模型检查",
  task_planning: "理解研究目标",
  dynamic_schema_planning: "生成动态字段",
  source_discovery: "发现数据来源",
  multi_source_search_planning: "规划多源检索",
  multi_source_search: "执行多源检索",
  source_selection: "筛选来源",
  source_triage: "来源分诊",
  multi_source_ingestion: "获取资料",
  arxiv_pdf_ingestion: "获取论文正文",
  artifact_action_planning: "规划资料操作",
  artifact_action_execution: "解析资料",
  artifact_search_more_source_selection: "扩展检索：筛选来源",
  artifact_search_more_source_triage: "扩展检索：来源分诊",
  artifact_search_more_ingestion: "扩展检索：获取资料",
  artifact_search_more_arxiv_ingestion: "扩展检索：下载论文",
  source_parsing: "解析正文与表格",
  figure_chart_extraction: "识别图表",
  section_interpretation: "理解论文结构",
  dynamic_extraction: "抽取动态记录",
  record_extraction: "抽取科研指标",
  normalization: "对齐字段与单位",
  provenance_tracking: "绑定来源证据",
  quality_validation: "质量校验",
  quality_validation_before_artifact_followup: "阶段性质量校验",
  export: "生成导出文件",
  completed: "任务完成",
  failed: "任务失败",
  cancelled: "任务已取消",
};

type ProgressBounds = readonly [start: number, end: number];

const STAGE_PROGRESS_BOUNDS: Record<string, ProgressBounds> = {
  ensure_llm_ready: [2, 4],
  task_planning: [4, 8],
  dynamic_schema_planning: [8, 12],
  source_discovery: [12, 16],
  multi_source_search_planning: [16, 20],
  multi_source_search: [20, 26],
  source_selection: [26, 32],
  source_triage: [32, 35],
  multi_source_ingestion: [35, 38],
  arxiv_pdf_ingestion: [38, 43],
  artifact_action_planning: [43, 47],
  artifact_action_execution: [47, 52],
  artifact_search_more_source_selection: [52, 54],
  artifact_search_more_source_triage: [54, 55],
  artifact_search_more_ingestion: [55, 56],
  artifact_search_more_arxiv_ingestion: [56, 58],
  source_parsing: [58, 64],
  figure_chart_extraction: [64, 70],
  section_interpretation: [70, 74],
  dynamic_extraction: [74, 80],
  record_extraction: [80, 84],
  normalization: [84, 88],
  provenance_tracking: [88, 92],
  quality_validation_before_artifact_followup: [92, 95],
  quality_validation: [95, 98],
  export: [98, 99],
};

export const STATUS_LABELS: Record<TaskStatus, string> = {
  partial: "部分完成",
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

export function stageLabel(stage?: string | null): string {
  if (!stage) return "等待状态";
  return STAGE_LABELS[stage] ?? stage.replaceAll("_", " ");
}

export function progressPercent(
  status: TaskStatus,
  stage?: string | null,
  progress?: TaskProgress | null,
  stageStatus?: string | null,
): number {
  if (status === "completed" || status === "partial") return 100;
  if (status === "cancelled") return 0;
  const bounds = stage ? STAGE_PROGRESS_BOUNDS[stage] : undefined;
  if (!bounds) {
    if (progress?.current != null && progress.total) {
      return Math.max(2, Math.min(98, Math.round((progress.current / progress.total) * 100)));
    }
    return status === "running" ? 4 : 1;
  }

  // API progress_index/progress_total describes work inside the current stage.
  // Place that fraction within the stage's slice of the whole pipeline instead
  // of presenting (for example) figure 1/12 as 8% overall progress.
  let stageFraction = stageStatus === "completed" ? 1 : 0;
  if (progress?.current != null && progress.total && progress.total > 0) {
    stageFraction = Math.max(0, Math.min(1, progress.current / progress.total));
  }
  const overall = bounds[0] + (bounds[1] - bounds[0]) * stageFraction;
  return Math.max(4, Math.min(98, Math.round(overall)));
}

export function overallProgressPercent(
  status: TaskStatus,
  events: TaskEvent[] | undefined,
  stage?: string | null,
  progress?: TaskProgress | null,
  stageStatus?: string | null,
): number {
  if (status === "completed" || status === "partial") return 100;
  if (status === "cancelled") return 0;

  let highest = status === "running" ? 2 : 1;
  for (const event of events ?? []) {
    const bounds = event.step ? STAGE_PROGRESS_BOUNDS[event.step] : undefined;
    if (!bounds) continue;
    const eventProgress = taskProgressFromEvent(event);
    highest = Math.max(
      highest,
      boundedStageProgress(bounds, event.status, eventProgress),
    );
  }

  return Math.max(highest, progressPercent(status, stage, progress, stageStatus));
}

function taskProgressFromEvent(event: TaskEvent): TaskProgress | null {
  const data = event.data;
  if (!data) return null;
  const current = numericValue(data.progress_index ?? data.index);
  const total = numericValue(data.progress_total ?? data.total);
  return current !== null && total !== null ? { current, total } : null;
}

function boundedStageProgress(
  bounds: ProgressBounds,
  stageStatus?: string | null,
  progress?: TaskProgress | null,
): number {
  let fraction = stageStatus === "completed" ? 1 : 0;
  if (progress?.current != null && progress.total && progress.total > 0) {
    fraction = Math.max(0, Math.min(1, progress.current / progress.total));
  }
  return Math.round(bounds[0] + (bounds[1] - bounds[0]) * fraction);
}

function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  const amount = value / 1024 ** index;
  return `${amount >= 10 || index === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`;
}

export function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
