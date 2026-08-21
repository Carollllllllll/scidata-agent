import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, cancelTask, getTask, getTaskEvents, retryTask } from "../../api/client";
import { Icon, type IconName } from "../../components/Icon";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDate, overallProgressPercent, stageLabel } from "../../lib/task";
import type { EvidenceRecord, TaskEvent, TaskResponse } from "../../types/api";
import {
  ChartsPanel,
  DynamicDataPanel,
  EvidenceDrawer,
  ExportsPanel,
  OverviewPanel,
  QualityPanel,
  SourcesPanel,
} from "./result/ResultPanels";

type TabId = "overview" | "sources" | "data" | "charts" | "quality" | "exports" | "events";

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
  const issueCount = task.quality_report?.issue_count ?? task.quality_report?.warning_count ?? 0;
  const tabs: TabDefinition[] = [
    { id: "overview", label: "总览", icon: "grid" },
    { id: "sources", label: "来源", icon: "database", count: sourceCount },
    { id: "data", label: "数据", icon: "table", count: recordCount },
    { id: "charts", label: "图像 / 图表", icon: "chart", count: chartCount },
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
    ?? (cancelMutation.error instanceof ApiError ? cancelMutation.error.message : cancelMutation.isError ? "取消任务失败，请稍后重试。" : null)
    ?? (retryMutation.error instanceof ApiError ? retryMutation.error.message : retryMutation.isError ? "重新运行失败，请稍后重试。" : null);

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
          {(task.status === "failed" || task.status === "cancelled") && <button className="text-button" type="button" disabled={retryMutation.isPending} onClick={() => retryMutation.mutate()}><Icon name="refresh" size={16} />重新运行</button>}
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
  const computedPercent = overallProgressPercent(
    task.status,
    events,
    task.current_step,
    task.progress,
    eventStatus,
  );
  const stableProgress = useRef({ taskId: task.task_id, percent: computedPercent });
  if (stableProgress.current.taskId !== task.task_id) {
    stableProgress.current = { taskId: task.task_id, percent: computedPercent };
  } else if (task.status === "cancelled") {
    stableProgress.current.percent = 0;
  } else {
    stableProgress.current.percent = Math.max(stableProgress.current.percent, computedPercent);
  }
  const percent = task.status === "completed" ? 100 : stableProgress.current.percent;
  const milestones = [
    { start: 0, end: 12, label: "规划" },
    { start: 12, end: 58, label: "来源" },
    { start: 58, end: 92, label: "解析" },
    { start: 92, end: 98, label: "校验" },
    { start: 98, end: 100, label: "导出" },
  ];

  return (
    <div className="progress-card">
      <div className="progress-card-top"><span>总体进度</span><strong>{percent}%</strong></div>
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
