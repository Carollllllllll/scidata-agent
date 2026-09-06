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
  partial: "部分完成",
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

// 这些值来自后端枚举、动态 Schema 或运行时状态。保留未知值的原样展示，
// 既方便排障，也避免将论文标题、来源名等原始科研数据误译。
const UI_VALUE_LABELS: Record<string, string> = {
  high: "高优先级",
  medium: "中优先级",
  low: "低优先级",
  info: "提示",
  warning: "警告",
  error: "错误",
  success: "成功",
  failed: "失败",
  completed: "已完成",
  partial: "部分完成",
  pending: "待处理",
  running: "运行中",
  queued: "排队中",
  unknown: "未知",
  resolved: "已定位",
  unresolved: "未定位",
  sufficient: "证据充分",
  exhausted: "检索已穷尽",
  covered: "已覆盖",
  supported: "已互证",
  approved: "确认通过",
  rejected: "已拒绝",
  needs_changes: "需要修改",
  string: "文本",
  number: "数值",
  integer: "整数",
  boolean: "布尔值",
  object: "对象",
  array: "列表",
  scientific_metrics: "科研指标",
  perovskite_materials: "钙钛矿材料",
  fabrication_methods: "制备方法",
  performance_metrics: "性能指标",
  stability_data: "稳定性数据",
  device_architecture: "器件结构",
  encapsulation_and_protection: "封装与防护",
  other_required_fields: "其他必填字段",
  paper_metadata: "论文元数据",
  material_composition: "材料组成",
  additives_used: "添加剂",
  crystal_structure_phase: "晶体结构/相态",
  deposition_method: "沉积方法",
  annealing_temperature: "退火温度",
  annealing_duration: "退火时长",
  solvent_system: "溶剂体系",
  pce_value: "PCE",
  jsc_value: "短路电流密度（Jsc）",
  voc_value: "开路电压（Voc）",
  fill_factor_value: "填充因子（FF）",
  test_condition: "测试条件",
  stability_duration: "稳定时长",
  final_pce_retention: "最终 PCE 保持率",
  testing_environment: "测试环境",
  light_intensity: "光照强度",
  temperature: "温度",
  substrate_type: "基底类型",
  electron_transport_layer: "电子传输层",
  hole_transport_layer: "空穴传输层",
  electrode_material: "电极材料",
  encapsulation_method: "封装方法",
  hermeticity_level: "密封等级",
  barrier_thickness: "阻隔层厚度",
  paper_title: "论文标题",
  method: "方法",
  experimental_result: "实验结果",
  table: "表格",
  figure: "图像/图表",
  text: "正文文本",
  text_extraction: "文本抽取",
  "text extraction": "文本抽取",
  table_extraction: "表格抽取",
  "table extraction": "表格抽取",
  figure_extraction: "图像/图表抽取",
  "figure extraction": "图像/图表抽取",
  chart_extraction: "图表数据抽取",
  "chart extraction": "图表数据抽取",
  pdf: "PDF 文档",
  material: "材料",
  metric_name: "指标名称",
  metric_value: "指标数值",
  unit: "单位",
  condition: "条件",
};

export function uiLabel(value?: string | null): string {
  if (!value) return "—";
  return UI_VALUE_LABELS[value] ?? value.replaceAll("_", " ");
}

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
  if (status === "completed") return 100;
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
  coverageScore?: number | null,
): number {
  if (status === "completed") return 100;
  if (status === "cancelled") return 0;
  if (status === "partial" && coverageScore != null && Number.isFinite(coverageScore)) {
    return Math.max(0, Math.min(99, Math.round(coverageScore * 100)));
  }

  // Dynamic runs may return to source discovery after parsing or validation.
  // Prefer the current recognized stage instead of freezing the historical
  // maximum, otherwise a reopened search appears already completed.
  if (stage && STAGE_PROGRESS_BOUNDS[stage]) {
    return progressPercent(status, stage, progress, stageStatus);
  }
  for (const event of [...(events ?? [])].reverse()) {
    const bounds = event.step ? STAGE_PROGRESS_BOUNDS[event.step] : undefined;
    if (!bounds) continue;
    const eventProgress = taskProgressFromEvent(event);
    return boundedStageProgress(bounds, event.status, eventProgress);
  }

  return progressPercent(status, stage, progress, stageStatus);
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
