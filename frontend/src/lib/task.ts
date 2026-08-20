import type { TaskProgress, TaskStatus } from "../types/api";

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
  source_parsing: "解析正文与表格",
  figure_chart_extraction: "识别图表",
  section_interpretation: "理解论文结构",
  dynamic_extraction: "抽取动态记录",
  record_extraction: "抽取科研指标",
  normalization: "对齐字段与单位",
  provenance_tracking: "绑定来源证据",
  quality_validation: "质量校验",
  export: "生成导出文件",
  completed: "任务完成",
  failed: "任务失败",
  cancelled: "任务已取消",
};

export const STATUS_LABELS: Record<TaskStatus, string> = {
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
): number {
  if (status === "completed") return 100;
  if (progress?.current != null && progress.total) {
    return Math.max(2, Math.min(98, Math.round((progress.current / progress.total) * 100)));
  }
  const index = stage ? AGENT_STAGES.indexOf(stage as (typeof AGENT_STAGES)[number]) : -1;
  if (status === "failed" && index < 0) return 4;
  if (status === "cancelled") return 0;
  if (index < 0) return status === "running" ? 4 : 1;
  return Math.max(4, Math.min(96, Math.round(((index + 1) / AGENT_STAGES.length) * 100)));
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
