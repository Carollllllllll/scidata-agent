import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

  import { ApiError, cancelTask, getTask, getTaskEvents, resumeTask, retryTask } from "../../api/client";
import { Icon, type IconName } from "../../components/Icon";
import { QualityBadge, StatusBadge } from "../../components/StatusBadge";
import { formatDate, overallProgressPercent, stageLabel } from "../../lib/task";
import type { EvidenceRecord, TaskEvent, TaskResponse } from "../../types/api";
import {
  ChartsPanel,
  DynamicDataPanel,
  EvidencePanel,
  EvidenceDrawer,
  ExportsPanel,
  OverviewPanel,
  QualityPanel,
  ReviewQueuePanel,
  SourcesPanel,
} from "./result/ResultPanels";

type TabId = "overview" | "sources" | "data" | "charts" | "evidence" | "review" | "quality" | "exports" | "events";

interface TabDefinition {
  id: TabId;
  label: string;
  icon: IconName;
  count?: number;
}

export function TaskWorkspace() {
  const { taskId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [selectedRecord, setSelectedRecord] = useState<EvidenceRecord | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState<string | null>(null);
  const copyResetTimer = useRef<number | null>(null);

  useEffect(() => {
    setSelectedRecord(null);
    setActiveTab("overview");
    setCopyError(null);
  }, [taskId]);

  useEffect(() => () => {
    if (copyResetTimer.current !== null) window.clearTimeout(copyResetTimer.current);
  }, []);

  const taskQuery = useQuery({
    queryKey: ["task", taskId],
    queryFn: () => getTask(taskId),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status !== "queued" && status !== "running") return false;
      return typeof document !== "undefined" && document.hidden ? false : 2_000;
    },
  });
  const progressEventsQuery = useQuery({
    queryKey: ["task-progress-events", taskId],
    queryFn: () => getTaskEvents(taskId, 120),
    enabled: Boolean(taskId),
    refetchInterval: () => {
      const status = taskQuery.data?.status;
      if (status !== "queued" && status !== "running") return false;
      return typeof document !== "undefined" && document.hidden ? false : 4_000;
    },
  });
  const cancelMutation = useMutation({
    mutationFn: () => cancelTask(taskId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["task", taskId] }),
        queryClient.invalidateQueries({ queryKey: ["task-progress-events", taskId] }),
        queryClient.invalidateQueries({ queryKey: ["tasks"] }),
      ]);
    },
  });
  const retryMutation = useMutation({
    mutationFn: () => retryTask(taskId),
    onSuccess: async (task) => {
      await queryClient.invalidateQueries({ queryKey: ["tasks"] });
      navigate(`/tasks/${task.task_id}`);
    },
  });
  const resumeMutation = useMutation({
    mutationFn: () => resumeTask(taskId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["task", taskId] }),
        queryClient.invalidateQueries({ queryKey: ["task-progress-events", taskId] }),
        queryClient.invalidateQueries({ queryKey: ["tasks"] }),
      ]);
    },
  });

  if (taskQuery.isLoading) return <TaskLoading />;
  if (taskQuery.isError || !taskQuery.data) {
    const message = taskQuery.error instanceof ApiError ? taskQuery.error.message : "任务信息加载失败。";
    return <TaskError message={message} retry={() => taskQuery.refetch()} />;
  }

  const task = taskQuery.data;
  const result = task.result;
  const recordCount = result?.dynamic_records?.length ?? 0;
  const sourceCount = result?.source_catalog?.length ?? 0;
  const chartCount = result?.chart_extractions?.length ?? 0;
  const evidenceCount = result?.evidence_traces?.length ?? 0;
  const reviewCount = result?.review_queue?.length ?? result?.needs_review_records?.length ?? 0;
  const coverageGaps = result?.coverage_report?.missing_requirements?.length ?? 0;
  const issueCount = Math.max(
    task.quality_report?.issue_count ?? task.quality_report?.warning_count ?? 0,
    coverageGaps,
  );
  const tabs: TabDefinition[] = [
    { id: "overview", label: "总览", icon: "grid" },
    { id: "sources", label: "来源", icon: "database", count: sourceCount },
    { id: "data", label: "数据", icon: "table", count: recordCount },
    { id: "charts", label: "图像 / 图表", icon: "chart", count: chartCount },
    { id: "evidence", label: "Evidence", icon: "link", count: evidenceCount },
    { id: "review", label: "复核", icon: "warning", count: reviewCount },
    { id: "quality", label: "质量", icon: "shield", count: issueCount },
    { id: "exports", label: "导出", icon: "download", count: Object.keys(task.download_urls).length },
    { id: "events", label: "运行记录", icon: "list" },
  ];

  async function copyTaskId() {
    try {
      await navigator.clipboard.writeText(task.task_id);
      setCopyError(null);
      setCopied(true);
      if (copyResetTimer.current !== null) window.clearTimeout(copyResetTimer.current);
      copyResetTimer.current = window.setTimeout(() => {
        setCopied(false);
        copyResetTimer.current = null;
      }, 1600);
    } catch {
      setCopied(false);
      setCopyError("复制失败，请手动选择任务 ID。浏览器可能未授予剪贴板权限。");
    }
  }

  const actionError = copyError
    ?? (cancelMutation.error instanceof ApiError ? cancelMutation.error.message : cancelMutation.isError ? "Cancel failed; please retry." : null)
    ?? (retryMutation.error instanceof ApiError ? retryMutation.error.message : retryMutation.isError ? "New-task retry failed; please retry." : null)
    ?? (resumeMutation.error instanceof ApiError ? resumeMutation.error.message : resumeMutation.isError ? "Checkpoint resume failed; inspect the run log." : null);
  /*
  const actionError = copyError
    ?? (cancelMutation.error instanceof ApiError ? cancelMutation.error.message : cancelMutation.isError ? "取消任务失败，请稍后重试。" : null)
    ?? (retryMutation.error instanceof ApiError ? retryMutation.error.message : retryMutation.isError ? "重新运行失败，请稍后重试。" : null);
    ?? (resumeMutation.error instanceof ApiError ? resumeMutation.error.message : resumeMutation.isError ? "从检查点恢复失败，请查看运行记录。" : null);

  */
  return (
    <div className="task-page">
      <div className="task-toolbar">
        <Link to="/" className="back-link"><span><Icon name="chevron" size={16} /></span>返回工作台</Link>
        <div className="task-toolbar-actions">
          <button className="text-button" type="button" onClick={() => taskQuery.refetch()} disabled={taskQuery.isFetching}>
            <Icon name="refresh" size={16} className={taskQuery.isFetching ? "spin" : ""} />刷新
          </button>
          <button className="text-button" type="button" onClick={copyTaskId}>
            <Icon name={copied ? "check" : "link"} size={16} />{copied ? "已复制" : "复制任务 ID"}
          </button>
          {task.status === "queued" && <button className="text-button" type="button" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate()}><Icon name="close" size={16} />取消任务</button>}
          {(task.status === "failed" || task.status === "cancelled") && <>
            <button className="text-button" type="button" disabled={resumeMutation.isPending || retryMutation.isPending} onClick={() => resumeMutation.mutate()}><Icon name="play" size={16} />从检查点继续</button>
            <button className="text-button" type="button" disabled={resumeMutation.isPending || retryMutation.isPending} onClick={() => retryMutation.mutate()}><Icon name="refresh" size={16} />新任务重跑</button>
          </>}
        </div>
      </div>

      {actionError && (
        <div className="failure-banner action-failure">
          <Icon name="warning" size={18} />
          <div><strong>操作未完成</strong><p>{actionError}</p></div>
        </div>
      )}

      <section className="task-hero">
        <div className="task-hero-main">
          <div className="task-identity">
            <StatusBadge status={task.status} />
            <span className="task-id">{task.task_id}</span>
            <span className="task-time"><Icon name="clock" size={14} /> {formatDate(task.created_at)}</span>
          </div>
          <h1>{task.research_question || "未命名科研任务"}</h1>
          <p className="task-message">{task.message || stageLabel(task.current_step)}</p>
        </div>
        <TaskProgressCard task={task} events={progressEventsQuery.data?.events} />
      </section>

      <LiveAgentRuntimePanel task={task} />

      {task.status === "failed" && (
        <div className="failure-banner">
          <Icon name="warning" size={20} />
          <div><strong>任务在「{stageLabel(task.current_step)}」阶段失败</strong><p>{task.error?.message || task.message || "后端未返回详细错误。"}</p></div>
          <button type="button" onClick={() => setActiveTab("events")}>查看运行记录</button>
        </div>
      )}

      <nav className="result-tabs" aria-label="任务结果">
        {tabs.map((tab) => (
          <button key={tab.id} type="button" className={activeTab === tab.id ? "active" : ""} onClick={() => setActiveTab(tab.id)}>
            <Icon name={tab.icon} size={17} />
            {tab.label}
            {tab.count !== undefined && tab.count > 0 && <span>{tab.count}</span>}
          </button>
        ))}
      </nav>

      <section className="result-content">
        {activeTab === "overview" && <OverviewPanel task={task} onSelectRecord={setSelectedRecord} />}
        {activeTab === "sources" && <SourcesPanel result={result} onSelectRecord={setSelectedRecord} />}
        {activeTab === "data" && <DynamicDataPanel result={result} onSelectRecord={setSelectedRecord} />}
        {activeTab === "charts" && <ChartsPanel result={result} />}
        {activeTab === "evidence" && <EvidencePanel result={result} onSelectRecord={setSelectedRecord} />}
        {activeTab === "review" && <ReviewQueuePanel task={task} onSelectRecord={setSelectedRecord} />}
        {activeTab === "quality" && <QualityPanel task={task} onSelectRecord={setSelectedRecord} />}
        {activeTab === "exports" && <ExportsPanel task={task} />}
        {activeTab === "events" && <EventsPanel taskId={taskId} status={task.status} />}
      </section>

      {selectedRecord && (
        <EvidenceDrawer
          record={selectedRecord}
          taskId={task.task_id}
          reviewDecision={task.review_decisions?.[selectedRecord.record_id]}
          sources={result?.source_catalog ?? []}
          onClose={() => setSelectedRecord(null)}
        />
      )}
    </div>
  );
}

function TaskProgressCard({ task, events }: { task: TaskResponse; events?: TaskEvent[] }) {
  const eventStatus = typeof task.event?.status === "string" ? task.event.status : null;
  const coverageScore =
    task.coverage?.coverage_score
    ?? task.coverage_report?.coverage_score
    ?? task.result?.coverage_report?.coverage_score
    ?? null;
  const percent = overallProgressPercent(
    task.status,
    events,
    task.current_step,
    task.progress,
    eventStatus,
    coverageScore,
  );
  const progressLabel = task.status === "partial" && coverageScore != null
    ? "需求覆盖率"
    : "工作流进度";
  const milestones = [
    { start: 0, end: 12, label: "规划" },
    { start: 12, end: 58, label: "来源" },
    { start: 58, end: 92, label: "解析" },
    { start: 92, end: 98, label: "校验" },
    { start: 98, end: 100, label: "导出" },
  ];

  return (
    <div className="progress-card">
      <div className="progress-card-top"><span>{progressLabel}</span><strong>{percent}%</strong></div>
      <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
      <div className="milestone-row">
        {milestones.map((milestone) => {
          const done = task.status === "completed" || percent >= milestone.end;
          const active = task.status === "running" && percent >= milestone.start && percent < milestone.end;
          return <span key={milestone.label} className={done ? "done" : active ? "active" : ""}><i>{done ? <Icon name="check" size={11} /> : null}</i>{milestone.label}</span>;
        })}
      </div>
      <div className="current-stage"><span className={task.status === "running" ? "pulse-dot" : "static-dot"} /><div><small>当前阶段</small><strong>{stageLabel(task.current_step)}</strong></div></div>
    </div>
  );
}

function LiveAgentRuntimePanel({ task }: { task: TaskResponse }) {
  const result = task.result;
  const resultDecisions = result?.agent_decision_history ?? [];
  const resultToolResults = result?.tool_result_history ?? [];
  const resultCatalog = result?.source_catalog ?? [];
  const runtime = task.runtime ?? (result ? {
    iteration: result.runtime_iteration,
    iteration_budget: result.runtime_iteration_budget,
    status: result.runtime_status,
    phase: result.runtime_phase,
    stop_reason: result.runtime_stop_reason,
    no_progress_streak: result.runtime_no_progress_streak,
    no_progress_limit: result.runtime_no_progress_limit,
    last_progress_iteration: result.runtime_last_progress_iteration,
    decision_count: resultDecisions.length,
    tool_result_count: resultToolResults.length,
    trace_count: result.agent_trace?.length ?? 0,
    recent_decisions: resultDecisions.slice(-3),
    recent_tool_results: resultToolResults.slice(-5),
    stop_rejections: result.stop_rejections?.slice(-8) ?? [],
  } : undefined);
  const coverage = task.coverage ?? result?.coverage_report;
  const sourceStatus = task.source_status ?? (result ? {
    catalog_count: resultCatalog.length,
    artifact_count: resultCatalog.reduce((total, source) => total + (source.artifacts?.length ?? 0), 0),
    source_status_counts: resultCatalog.reduce<Record<string, number>>((counts, source) => {
      const status = source.status || "unknown";
      counts[status] = (counts[status] ?? 0) + 1;
      return counts;
    }, {}),
  } : undefined);
  if (!runtime && !coverage && !sourceStatus) return null;

  const decisions = runtime?.recent_decisions?.length ? runtime.recent_decisions : resultDecisions.slice(-3);
  const latestDecision = decisions.length ? decisions[decisions.length - 1] : undefined;
  const calls = latestDecision?.tool_calls ?? [];
  const recentResults = runtime?.recent_tool_results?.length ? runtime.recent_tool_results : resultToolResults.slice(-5);
  const connectors = sourceStatus?.connectors ?? [];
  const stopReason = runtime?.stop_reason ?? (
    task.status === "running"
      ? "Agent is still working; no stop reason has been recorded."
      : task.message ?? "No stop reason recorded."
  );

  return (
    <section className="live-runtime-panel panel-card" aria-live="polite">
      <div className="live-runtime-heading">
        <div><span className="eyebrow dark">LIVE AGENT RUNTIME</span><h2>Current decision state</h2></div>
        <QualityBadge tone={runtime?.status === "completed" ? "success" : runtime?.status === "partial" ? "warning" : "info"}>
          {runtime?.status || task.status}
        </QualityBadge>
      </div>
      <div className="live-runtime-stats">
        <span><small>Iteration</small><strong>{runtime?.iteration ?? 0}{runtime?.iteration_budget ? ` / ${runtime.iteration_budget}` : ""}</strong></span>
        <span><small>Decisions</small><strong>{runtime?.decision_count ?? decisions.length}</strong></span>
        <span><small>Tools</small><strong>{runtime?.tool_result_count ?? recentResults.length}</strong></span>
        <span><small>Coverage</small><strong>{coverage?.coverage_score !== undefined ? `${Math.round(coverage.coverage_score * 100)}%` : "-"}</strong></span>
        <span><small>Sources</small><strong>{sourceStatus?.catalog_count ?? resultCatalog.length}</strong></span>
      </div>
      <div className="live-runtime-columns">
        <div className="live-runtime-decision">
          <small>Latest model decision</small>
          <strong>{latestDecision?.decision || (task.status === "running" ? "Waiting for the next decision" : "No decision recorded")}</strong>
          {latestDecision?.reason && <p>{latestDecision.reason}</p>}
          {calls.length > 0 && <div className="live-runtime-tools">{calls.slice(0, 8).map((call) => <span key={call.call_id}><Icon name="play" size={12} />{call.tool_name}</span>)}</div>}
        </div>
        <div className="live-runtime-sources">
          <small>Source status</small>
          <div className="live-runtime-status-list">
            {Object.entries(sourceStatus?.source_status_counts ?? {}).slice(0, 5).map(([name, count]) => <span key={name}>{name}<strong>{count}</strong></span>)}
            {connectors.slice(-4).map((connector, index) => <span key={`${connector.connector || connector.connector_name || "connector"}-${index}`}><em>{connector.connector || connector.connector_name || "connector"}</em><strong>{connector.status || "unknown"}</strong></span>)}
          </div>
        </div>
      </div>
      {runtime?.latest_event?.event_type && <div className="live-runtime-latest"><small>Latest runtime event</small><span><strong>{runtime.latest_event.event_type.replaceAll("_", " ")}</strong>{runtime.latest_event.tool_name ? ` / ${runtime.latest_event.tool_name}` : ""}{runtime.latest_event.status ? ` / ${runtime.latest_event.status}` : ""}</span></div>}
      {recentResults.length > 0 && <div className="live-runtime-results"><small>Recent tool results</small><div>{recentResults.slice(-5).map((result) => <span key={`${result.call_id}-${result.tool_name}`} className={`live-tool-result live-tool-${result.status}`}><strong>{result.tool_name}</strong><em>{result.status}</em>{(result.evidence_refs?.length ?? 0) > 0 && <i>{result.evidence_refs?.length} evidence</i>}</span>)}</div></div>}
      {(coverage?.missing_requirements?.length ?? 0) > 0 && <div className="live-runtime-gap"><Icon name="warning" size={14} /><span>Open requirements: {coverage?.missing_requirements?.slice(0, 4).join(", ")}</span></div>}
      {stopReason && <div className="live-runtime-stop"><small>Stop reason</small><span>{stopReason}</span></div>}
      {runtime?.phase && <div className="live-runtime-latest"><small>Runtime phase</small><span>{runtime.phase.replaceAll("_", " ")}{runtime.no_progress_streak !== undefined && runtime.no_progress_limit !== undefined ? `; no progress ${runtime.no_progress_streak}/${runtime.no_progress_limit}` : ""}</span></div>}
      {(runtime?.stop_rejections?.length ?? 0) > 0 && <div className="live-runtime-gap"><Icon name="warning" size={14} /><span>Policy / stop-gate rejections: {runtime?.stop_rejections?.slice(-2).join("; ")}</span></div>}
    </section>
  );
}

function EventsPanel({ taskId, status }: { taskId: string; status: TaskResponse["status"] }) {
  const events = useQuery({
    queryKey: ["task-events", taskId],
    queryFn: () => getTaskEvents(taskId, 80),
    refetchInterval: () => {
      if (status !== "queued" && status !== "running") return false;
      return typeof document !== "undefined" && document.hidden ? false : 4_000;
    },
  });

  return (
    <div className="panel-card events-panel">
      <div className="panel-heading"><div><span className="eyebrow dark">AUDIT TRAIL</span><h2>运行记录</h2><p>只读取最近 80 条事件，避免反复传输完整日志。</p></div><button className="icon-button" onClick={() => events.refetch()}><Icon name="refresh" size={16} /></button></div>
      {events.isLoading && <div className="inline-loading"><span className="spinner dark" /> 正在读取事件…</div>}
      {events.isError && <EmptyResult icon="warning" title="运行记录加载失败" text="任务结果仍可继续浏览，请稍后重试。" />}
      {events.data?.events.length === 0 && <EmptyResult icon="clock" title="还没有运行事件" text="任务进入执行阶段后，节点状态会出现在这里。" />}
      <div className="event-timeline">
        {events.data?.events.slice().reverse().map((event, index) => (
          <article key={`${event.timestamp}:${event.step}:${index}`}>
            <span className={`event-dot event-${event.status || "unknown"}`} />
            <div className="event-meta"><strong>{stageLabel(event.step)}</strong><time>{formatDate(event.timestamp)}</time></div>
            <p>{event.message || "无附加说明"}</p>
            {event.duration_ms !== undefined && <small>{event.duration_ms} ms</small>}
          </article>
        ))}
      </div>
    </div>
  );
}

function TaskLoading() {
  return <div className="task-loading"><span className="loader-orbit"><i /></span><h2>正在载入任务</h2><p>读取任务状态、结果和证据索引…</p></div>;
}

function TaskError({ message, retry }: { message: string; retry: () => void }) {
  return <div className="task-error"><span><Icon name="warning" size={26} /></span><h2>无法打开任务</h2><p>{message}</p><div><button className="secondary-button" onClick={retry}><Icon name="refresh" size={16} />重试</button><Link className="primary-button small" to="/">返回工作台</Link></div></div>;
}

function EmptyResult({ icon, title, text }: { icon: IconName; title: string; text: string }) {
  return <div className="empty-result"><span><Icon name={icon} size={23} /></span><strong>{title}</strong><p>{text}</p></div>;
}
