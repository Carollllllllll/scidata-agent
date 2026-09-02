import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  API_AUTH_ENABLED,
  ApiError,
  apiUrl,
  getAssetBlob,
  getTextAsset,
  openApiAsset,
  submitReview,
} from "../../../api/client";
import { Icon } from "../../../components/Icon";
import { QualityBadge } from "../../../components/StatusBadge";
import { displayValue, stageLabel } from "../../../lib/task";
import type {
  AgentResult,
  ChartCorrection,
  ChartExtraction,
  ChartValidation,
  CrossModalCheck,
  CoverageReport,
  DynamicFieldSpec,
  DynamicRecord,
  EvidenceRecord,
  EvidenceTrace,
  FigureAsset,
  SourceCatalogEntry,
  ReviewDecision,
  ReviewQueueItem,
  TaskResponse,
} from "../../../types/api";

export function ReviewQueuePanel({
  task,
  onSelectRecord,
}: {
  task: TaskResponse;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  const queryClient = useQueryClient();
  const items = task.result?.review_queue ?? [];
  const records = [...(task.result?.dynamic_records ?? []), ...(task.result?.records ?? [])];
  const mutation = useMutation({
    mutationFn: ({ item, decision }: { item: ReviewQueueItem; decision: ReviewDecision["decision"] }) =>
      submitReview(task.task_id, item.review_id, decision),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["task", task.task_id] });
    },
  });

  return (
    <section className="panel-card standalone-review-queue">
      <div className="panel-heading">
        <div><span className="eyebrow dark">HUMAN REVIEW</span><h2>主动复核队列</h2><p>每一项都指向已有审计证据；人工结论不会覆盖原始抽取值。</p></div>
        <QualityBadge tone={items.length ? "warning" : "success"}>{items.length}</QualityBadge>
      </div>
      {items.length === 0 ? (
        <EmptyState icon="check" title="当前没有待复核风险" text="自动审计未生成需要人工确认的队列项。" success />
      ) : (
        <div className="review-queue-grid">
          {items.map((item) => {
            const decision = task.review_decisions?.[item.review_id] ?? (item.record_id ? task.review_decisions?.[item.record_id] : undefined);
            const record = item.record_id ? records.find((candidate) => candidate.record_id === item.record_id) : undefined;
            return (
              <article className="review-queue-item" key={item.review_id}>
                <div className="review-queue-item-head">
                  <QualityBadge tone={item.priority === "high" ? "danger" : item.priority === "medium" ? "warning" : "info"}>{item.priority}</QualityBadge>
                  <span>{item.risk_type}</span>
                  {decision && <strong>{reviewDecisionLabel(decision.decision)}</strong>}
                </div>
                <h3>{item.title}</h3>
                <p>{item.reason}</p>
                <small>{item.source_file ?? item.subject_type}{item.page ? ` · page ${item.page}` : ""}</small>
                <div className="review-queue-actions">
                  {record && <button type="button" onClick={() => onSelectRecord(record)}>查看证据</button>}
                  <button type="button" disabled={mutation.isPending} onClick={() => mutation.mutate({ item, decision: "approved" })}>通过</button>
                  <button type="button" disabled={mutation.isPending} onClick={() => mutation.mutate({ item, decision: "needs_changes" })}>需修改</button>
                  <button type="button" disabled={mutation.isPending} onClick={() => mutation.mutate({ item, decision: "rejected" })}>拒绝</button>
                </div>
              </article>
            );
          })}
        </div>
      )}
      {mutation.isError && <p className="form-error">复核提交失败，请检查任务状态后重试。</p>}
    </section>
  );
}

export function OverviewPanel({
  task,
  onSelectRecord,
}: {
  task: TaskResponse;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  const result = task.result;
  const summary = task.summary;
  const quality = task.quality_report;

  if (!result) {
    return (
      <div className="live-state-grid">
        <div className="panel-card live-state-card">
          <span className="live-orbit"><i /></span>
          <div className="eyebrow dark">LIVE AGENT STATE</div>
          <h2>{task.status === "failed" ? "任务未生成可用结果" : stageLabel(task.current_step)}</h2>
          <p>{task.message || "Agent 正在准备任务上下文。"}</p>
          <div className="live-facts">
            <span><small>任务状态</small><strong>{task.status}</strong></span>
            <span><small>上传文件</small><strong>{task.uploads.length}</strong></span>
            <span><small>当前节点</small><strong>{task.current_step || "queued"}</strong></span>
          </div>
        </div>
        <div className="panel-card expectation-card">
          <div className="panel-heading"><div><h3>结果生成后将显示</h3><p>页面不会用模拟数据填补尚未产生的结果。</p></div></div>
          <ul>
            <li><Icon name="database" /> 数据来源与选择理由</li>
            <li><Icon name="table" /> 动态字段和清洗后记录</li>
            <li><Icon name="link" /> 页码、原文与证据链</li>
            <li><Icon name="shield" /> 警告、冲突与复核队列</li>
          </ul>
        </div>
      </div>
    );
  }

  const metrics = [
    { label: "发现来源", value: result.source_catalog?.length ?? 0, hint: "论文 / 数据库 / 附件", icon: "database" as const, tone: "blue" },
    { label: "结构化记录", value: result.dynamic_records?.length ?? summary?.dynamic_records_extracted ?? 0, hint: `${summary?.dynamic_tables_count ?? 0} 个动态表`, icon: "table" as const, tone: "violet" },
    { label: "证据文本", value: `${Math.round((quality?.evidence_text_coverage ?? quality?.evidence_coverage ?? 0) * 100)}%`, hint: "仅表示记录绑定了证据文本", icon: "link" as const, tone: "green" },
    { label: "需要复核", value: Math.max(result.needs_review_records?.length ?? 0, quality?.review_count ?? 0), hint: `${quality?.conflict_count ?? 0} 个跨来源冲突`, icon: "warning" as const, tone: "amber" },
  ];
  const tables = result.dynamic_extraction_plan?.dynamic_tables ?? [];
  const reviewRecords = result.needs_review_records ?? [];

  return (
    <div className="overview-layout">
      <div className="metric-grid">
        {metrics.map((metric) => (
          <article className={`metric-card metric-${metric.tone}`} key={metric.label}>
            <span className="metric-icon"><Icon name={metric.icon} size={20} /></span>
            <div><small>{metric.label}</small><strong>{metric.value}</strong><p>{metric.hint}</p></div>
          </article>
        ))}
      </div>

      <div className="overview-two-column">
        <section className="panel-card schema-panel">
          <div className="panel-heading">
            <div><span className="eyebrow dark">DYNAMIC SCHEMA</span><h2>本次任务的数据结构</h2><p>{result.dynamic_extraction_plan?.domain || "通用科研数据"} · {result.dynamic_extraction_plan?.task_type || "数据整合"}</p></div>
            <QualityBadge tone="info">{tables.length} 个表</QualityBadge>
          </div>
          {tables.length === 0 ? (
            <EmptyState icon="table" title="尚未生成动态结构" text="若任务在规划阶段失败，请在运行记录中查看原因。" />
          ) : (
            <div className="schema-list">
              {tables.map((table, index) => {
                const count = result.dynamic_records?.filter((record) => record.table_name === table.table_name).length ?? 0;
                return (
                  <article key={table.table_name}>
                    <span className="schema-index">{String(index + 1).padStart(2, "0")}</span>
                    <div><strong>{humanize(table.table_name)}</strong><p>{table.description || `${table.entity_type} 数据表`}</p><div>{table.fields.slice(0, 5).map((field) => <span key={field.name}>{field.name}</span>)}{table.fields.length > 5 && <span>+{table.fields.length - 5}</span>}</div></div>
                    <span className="schema-count">{count}<small>记录</small></span>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="panel-card quality-snapshot">
          <div className="panel-heading"><div><span className="eyebrow dark">QUALITY SNAPSHOT</span><h2>可信度快照</h2><p>证据覆盖与异常不会被折叠隐藏。</p></div></div>
          <CoverageRow label="证据文本存在率" value={quality?.evidence_text_coverage ?? quality?.evidence_coverage ?? 0} tone="blue" />
          <CoverageRow label="数值证据覆盖" value={quality?.value_evidence_coverage ?? 0} tone="green" />
          <CoverageRow label="PDF 页码验证率" value={quality?.provenance_page_coverage ?? 0} tone="violet" />
          <div className="quality-numbers">
            <span><strong>{quality?.record_count ?? 0}</strong><small>记录总数</small></span>
            <span><strong>{quality?.warning_count ?? 0}</strong><small>警告</small></span>
            <span><strong>{quality?.error_count ?? 0}</strong><small>错误</small></span>
            <span><strong>{quality?.conflict_count ?? 0}</strong><small>冲突</small></span>
          </div>
          {quality?.notes?.slice(0, 2).map((note) => <p className="quality-note" key={note}><Icon name="info" size={15} />{note}</p>)}
        </section>
      </div>

      <div className="overview-two-column lower">
        <section className="panel-card result-preview">
          <div className="panel-heading"><div><span className="eyebrow dark">RECENT RECORDS</span><h2>结构化结果预览</h2></div><span>{result.dynamic_records?.length ?? 0} 条</span></div>
          {(result.dynamic_records?.length ?? 0) === 0 ? <EmptyState icon="table" title="没有可用记录" text="没有找到数据时保持为空，不生成占位结果。" /> : (
            <div className="preview-records">
              {result.dynamic_records?.slice(0, 5).map((record) => (
                <button type="button" key={record.record_id} onClick={() => onSelectRecord(record)}>
                  <span className="record-table">{humanize(record.table_name)}</span>
                  <span className="record-main"><strong>{recordTitle(record)}</strong><small>{record.source_file}{record.page ? ` · 第 ${record.page} 页` : ""}</small></span>
                  <Confidence value={record.confidence} />
                  <Icon name="chevron" size={15} />
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="panel-card review-preview">
          <div className="panel-heading"><div><span className="eyebrow dark">REVIEW QUEUE</span><h2>待复核记录</h2></div>{reviewRecords.length > 0 && <QualityBadge tone="warning">{reviewRecords.length}</QualityBadge>}</div>
          {reviewRecords.length === 0 ? <EmptyState icon="check" title="当前没有待复核记录" text="这不代表绝对正确；仍应结合覆盖率和来源质量判断。" success /> : (
            <div className="review-list">
              {reviewRecords.slice(0, 5).map((record) => (
                <button type="button" key={record.record_id} onClick={() => onSelectRecord(record)}>
                  <span><Icon name="warning" size={16} /></span><div><strong>{recordTitle(record)}</strong><p>{record.warnings[0] || "低置信度或证据不完整"}</p></div><Icon name="chevron" size={15} />
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export function DynamicDataPanel({
  result,
  onSelectRecord,
}: {
  result?: AgentResult | null;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  const cleanRecords = result?.dynamic_records ?? [];
  const rawRecords = result?.dynamic_records_raw ?? [];
  const [showRaw, setShowRaw] = useState(false);
  const records = showRaw ? rawRecords : cleanRecords;
  const specs = result?.dynamic_extraction_plan?.dynamic_tables ?? [];
  const tableNames = useMemo(
    () => Array.from(new Set([...specs.map((table) => table.table_name), ...records.map((record) => record.table_name)])),
    [records, specs],
  );
  const [selectedTable, setSelectedTable] = useState("all");
  const [search, setSearch] = useState("");
  const [onlyWarnings, setOnlyWarnings] = useState(false);
  const [page, setPage] = useState(0);
  const searchableRecords = useMemo(
    () => records.map((record) => ({ record, searchText: JSON.stringify(record).toLowerCase() })),
    [records],
  );

  useEffect(() => {
    if (selectedTable !== "all" && !tableNames.includes(selectedTable)) setSelectedTable("all");
  }, [selectedTable, tableNames]);

  const filtered = useMemo(() => searchableRecords.filter(({ record, searchText }) => {
    if (selectedTable !== "all" && record.table_name !== selectedTable) return false;
    if (onlyWarnings && record.warnings.length === 0) return false;
    if (!search.trim()) return true;
    const needle = search.toLowerCase();
    return searchText.includes(needle);
  }).map(({ record }) => record), [searchableRecords, selectedTable, onlyWarnings, search]);

  const activeSpec = specs.find((table) => table.table_name === selectedTable);
  const fields = useMemo<DynamicFieldSpec[]>(() => {
    if (activeSpec) return activeSpec.fields;
    const frequencies = new Map<string, number>();
    filtered.slice(0, 100).forEach((record) => Object.keys(record.fields).forEach((name) => {
      frequencies.set(name, (frequencies.get(name) ?? 0) + 1);
    }));
    const limit = selectedTable === "all" ? 4 : 12;
    return Array.from(frequencies.entries())
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, limit)
      .map(([name]) => ({ name, type: "unknown", required: false, evidence_required: true }));
  }, [activeSpec, filtered, selectedTable]);
  const pageSize = 20;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visible = filtered.slice(page * pageSize, (page + 1) * pageSize);

  useEffect(() => setPage(0), [selectedTable, onlyWarnings, search, showRaw]);

  if (!result) return <PanelEmptyState title="结果尚未生成" text="任务完成后，清洗记录和原始记录会在这里分开展示。" />;

  return (
    <div className="data-layout">
      <aside className="table-sidebar panel-card">
        <div className="table-sidebar-heading"><span className="eyebrow dark">DATA TABLES</span><h3>动态数据表</h3></div>
        <button type="button" className={selectedTable === "all" ? "active" : ""} onClick={() => setSelectedTable("all")}>
          <span><Icon name="layers" size={17} /><span><strong>全部记录</strong><small>跨表统一浏览</small></span></span><b>{records.length}</b>
        </button>
        {tableNames.map((name) => {
          const spec = specs.find((table) => table.table_name === name);
          const count = records.filter((record) => record.table_name === name).length;
          return <button type="button" key={name} className={selectedTable === name ? "active" : ""} onClick={() => setSelectedTable(name)}><span><Icon name="table" size={17} /><span><strong>{humanize(name)}</strong><small>{spec?.entity_type || "动态字段"}</small></span></span><b>{count}</b></button>;
        })}
        <div className="table-mode-note"><Icon name="info" size={15} /><span>字段由当前研究问题动态生成，不预设学科模板。</span></div>
      </aside>

      <section className="panel-card data-table-panel">
        <div className="data-toolbar">
          <div><span className="eyebrow dark">STRUCTURED RESULTS</span><h2>{selectedTable === "all" ? "全部结构化记录" : humanize(selectedTable)}</h2><p>{activeSpec?.description || `共 ${filtered.length} 条记录，点击任意行查看来源证据。`}</p></div>
          <div className="record-mode-switch"><button className={!showRaw ? "active" : ""} onClick={() => setShowRaw(false)}>清洗结果 <span>{cleanRecords.length}</span></button><button className={showRaw ? "active" : ""} onClick={() => setShowRaw(true)}>原始记录 <span>{rawRecords.length}</span></button></div>
        </div>
        <div className="table-controls">
          <label className="search-control"><Icon name="search" size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索字段、来源或证据" /></label>
          <button type="button" className={onlyWarnings ? "filter-active" : ""} onClick={() => setOnlyWarnings((value) => !value)}><Icon name="filter" size={16} />仅看警告</button>
          <span className="result-count">{filtered.length} 条</span>
        </div>

        {visible.length === 0 ? <EmptyState icon="table" title="没有符合条件的记录" text={records.length === 0 ? "Agent 没有抽取到可用数据，页面不会填充假结果。" : "请清除筛选条件后重试。"} /> : (
          <div className="dynamic-table-wrap">
            <table className="dynamic-table">
              <thead><tr><th className="sticky-col">记录</th>{fields.map((field) => <th key={field.name} title={field.description || undefined}>{humanize(field.name)}{field.required && <i>*</i>}<small>{field.type}</small></th>)}<th>来源</th><th>置信度</th><th>状态</th></tr></thead>
              <tbody>{visible.map((record) => <tr key={record.record_id} onClick={() => onSelectRecord(record)} tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelectRecord(record); } }}><td className="sticky-col"><strong>{record.record_id.slice(0, 12)}</strong><small>{humanize(record.table_name)}</small></td>{fields.map((field) => <td key={field.name}><span className="cell-value">{displayValue(record.fields[field.name])}</span></td>)}<td><span className="source-cell"><Icon name="document" size={14} />{record.source_file}<small>{record.page ? `第 ${record.page} 页` : record.source_type}</small></span></td><td><Confidence value={record.confidence} /></td><td>{record.warnings.length > 0 ? <QualityBadge tone="warning">{record.warnings.length} 警告</QualityBadge> : <QualityBadge tone="success">有证据</QualityBadge>}</td></tr>)}</tbody>
            </table>
          </div>
        )}
        {filtered.length > pageSize && <div className="pagination"><span>第 {page + 1} / {pageCount} 页</span><div><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>上一页</button><button disabled={page + 1 >= pageCount} onClick={() => setPage((value) => value + 1)}>下一页</button></div></div>}
      </section>
    </div>
  );
}

export function SourcesPanel({ result, onSelectRecord }: { result?: AgentResult | null; onSelectRecord: (record: EvidenceRecord) => void }) {
  const sources = result?.source_catalog ?? [];
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [selectedSource, setSelectedSource] = useState<SourceCatalogEntry | null>(null);
  const [page, setPage] = useState(0);
  const statuses = Array.from(new Set(sources.map((source) => source.status).filter(Boolean))) as string[];
  const filtered = sources.filter((source) => {
    if (status !== "all" && source.status !== status) return false;
    if (!search.trim()) return true;
    return `${source.title} ${source.provider} ${source.source_type}`.toLowerCase().includes(search.toLowerCase());
  });
  const selectedCount = sources.filter((source) => source.selection_action === "select" || ["selected", "downloaded", "parsed"].includes(source.status ?? "")).length;
  const downloadedCount = sources.filter((source) => ["downloaded", "parsed"].includes(source.status ?? "") || source.artifacts?.some((artifact) => ["downloaded", "parsed"].includes(artifact.status ?? ""))).length;
  const pageSize = 30;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const visibleSources = filtered.slice(page * pageSize, (page + 1) * pageSize);

  useEffect(() => setPage(0), [search, status]);
  useEffect(() => {
    if (page >= pageCount) setPage(Math.max(0, pageCount - 1));
  }, [page, pageCount]);

  if (!result) return <PanelEmptyState title="来源尚未生成" text="Agent 完成来源发现后，这里会展示选择理由、处理状态和关联资料。" />;

  return (
    <div className="sources-layout">
      <section className="panel-card sources-panel">
        <div className="panel-heading wide"><div><span className="eyebrow dark">SOURCE CATALOG</span><h2>数据来源目录</h2><p>来源生命周期与记录质量分开显示。</p></div><div className="source-summary"><span><strong>{sources.length}</strong><small>已发现</small></span><span><strong>{selectedCount}</strong><small>已选择</small></span><span><strong>{downloadedCount}</strong><small>已下载</small></span><span><strong>{sources.filter((source) => source.status === "parsed").length}</strong><small>已解析</small></span></div></div>
        <div className="table-controls">
          <label className="search-control"><Icon name="search" size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索标题、Provider 或类型" /></label>
          <select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">全部状态</option>{statuses.map((item) => <option key={item} value={item}>{sourceStatusLabel(item)}</option>)}</select>
          <span className="result-count">{filtered.length} 个来源</span>
        </div>
        {filtered.length === 0 ? <EmptyState icon="database" title="没有符合条件的来源" text={sources.length === 0 ? "本次任务没有发现来源，或在来源规划前失败。" : "请调整筛选条件。"} /> : (
          <div className="source-list">
            {visibleSources.map((source) => (
              <button type="button" key={source.source_id} className={selectedSource?.source_id === source.source_id ? "active" : ""} onClick={() => setSelectedSource(source)}>
                <span className={`source-type-icon source-${source.source_type || "unknown"}`}><Icon name={source.source_type === "open_database" || source.source_type === "dataset" ? "database" : source.source_type === "image" ? "chart" : "document"} size={19} /></span>
                <span className="source-main"><span><QualityBadge tone={sourceTone(source.status)}>{sourceStatusLabel(source.status)}</QualityBadge><small>{source.provider || source.source_type || "未知来源"}</small></span><strong>{source.title}</strong><p>{source.reason || source.failure_reason || "未提供选择理由"}</p></span>
                <span className="relevance"><small>相关性</small><strong>{Math.round((source.relevance_score ?? 0) * 100)}</strong></span>
                <Icon name="chevron" size={16} />
              </button>
            ))}
          </div>
        )}
        {filtered.length > pageSize && <div className="pagination"><span>第 {page + 1} / {pageCount} 页</span><div><button disabled={page === 0} onClick={() => setPage((value) => value - 1)}>上一页</button><button disabled={page + 1 >= pageCount} onClick={() => setPage((value) => value + 1)}>下一页</button></div></div>}
      </section>

      <aside className="panel-card source-detail">
        {selectedSource ? <SourceDetail source={selectedSource} records={recordsForSource(selectedSource, result.dynamic_records ?? [])} onSelectRecord={onSelectRecord} /> : <div className="source-detail-placeholder"><span><Icon name="database" size={24} /></span><h3>选择一个来源</h3><p>查看 Provider、选择理由、Artifact、下载状态和失败原因。</p></div>}
      </aside>

      {(result.connector_status?.length ?? 0) > 0 && (
        <section className="panel-card connector-panel">
          <div className="panel-heading"><div><span className="eyebrow dark">CONNECTORS</span><h2>连接器状态</h2></div></div>
          <div className="connector-grid">{result.connector_status?.map((connector, index) => <article key={`${String(connector.connector_name)}:${index}`}><span className={`service-indicator ${connector.status === "success" || connector.status === "completed" ? "online" : "offline"}`} /><div><strong>{displayValue(connector.connector_name ?? connector.provider ?? "Connector")}</strong><p>{displayValue(connector.message ?? connector.error ?? connector.status)}</p></div></article>)}</div>
        </section>
      )}
    </div>
  );
}

export function SourceDetail({ source, records, onSelectRecord }: { source: SourceCatalogEntry; records: DynamicRecord[]; onSelectRecord: (record: EvidenceRecord) => void }) {
  return (
    <>
      <div className="source-detail-head"><span className="section-icon"><Icon name="document" size={19} /></span><QualityBadge tone={sourceTone(source.status)}>{sourceStatusLabel(source.status)}</QualityBadge></div>
      <h2>{source.title}</h2>
      <div className="source-detail-meta"><span><small>Provider</small><strong>{source.provider || "—"}</strong></span><span><small>类型</small><strong>{source.source_type || "—"}</strong></span><span><small>相关性</small><strong>{Math.round((source.relevance_score ?? 0) * 100)}%</strong></span></div>
      {source.url && <a href={source.url} target="_blank" rel="noreferrer" className="source-link"><Icon name="external" size={15} />打开原始来源</a>}
      <div className="detail-block"><small>选择理由</small><p>{source.reason || "未提供选择理由。"}</p></div>
      {source.failure_reason && <div className="detail-warning"><Icon name="warning" size={16} /><span><strong>处理失败</strong><p>{source.failure_reason}</p></span></div>}
      <div className="detail-block"><small>关联资料</small>{(source.artifacts?.length ?? 0) === 0 ? <p>没有可读取的 Artifact。</p> : <div className="artifact-list">{source.artifacts?.map((artifact) => <article key={artifact.artifact_id}><span><Icon name="file" size={15} /></span><div><strong>{artifact.name || artifact.artifact_type || "artifact"}</strong><small>{sourceStatusLabel(artifact.status)}{artifact.relevance_score != null ? ` · 相关性 ${artifact.relevance_score.toFixed(1)}/4` : ""}</small>{artifact.evidence_types?.length ? <small>{artifact.evidence_types.join(" · ")}</small> : null}{artifact.relevance_reason ? <p className="artifact-reason">{artifact.relevance_reason}</p> : null}</div>{artifact.asset_url ? <ApiAssetLink path={artifact.asset_url} ariaLabel="打开关联资料"><Icon name="external" size={14} /></ApiAssetLink> : artifact.url ? <a href={artifact.url} target="_blank" rel="noreferrer"><Icon name="external" size={14} /></a> : null}</article>)}</div>}</div>
      <div className="detail-block"><small>关联记录</small>{records.length === 0 ? <p>当前没有可反向关联的结构化记录。</p> : <div className="source-record-links">{records.slice(0, 8).map((record) => <button type="button" key={record.record_id} onClick={() => onSelectRecord(record)}><span>{recordFieldSummary(record)}</span><small>{record.page ? `第 ${record.page} 页` : record.source_file}</small><Icon name="chevron" size={14} /></button>)}</div>}</div>
    </>
  );
}

export function ChartsPanel({ result }: { result?: AgentResult | null }) {
  const figures = result?.figures ?? [];
  const extractionByFigure = useMemo(() => new Map((result?.chart_extractions ?? []).map((item) => [item.figure_id, item])), [result?.chart_extractions]);
  const validationByFigure = useMemo(() => new Map((result?.chart_validations ?? []).map((item) => [item.figure_id, item])), [result?.chart_validations]);
  const correctionByFigure = useMemo(() => new Map((result?.chart_corrections ?? []).map((item) => [item.figure_id, item])), [result?.chart_corrections]);
  const crossModalBySubject = useMemo(() => new Map((result?.cross_modal_checks ?? []).map((item) => [item.subject_id, item])), [result?.cross_modal_checks]);

  if (!result) return <PanelEmptyState title="图表结果尚未生成" text="任务解析 PDF 图像后，图表和校验信息会出现在这里。" />;
  return (
    <section className="panel-card charts-panel">
      <div className="panel-heading wide"><div><span className="eyebrow dark">FIGURE INTELLIGENCE</span><h2>图表与图像解析</h2><p>近似读数会明确标记，无法确认的数字不会作为确定值展示。</p></div><div className="source-summary"><span><strong>{figures.length}</strong><small>检测图像</small></span><span><strong>{result.chart_extractions?.length ?? 0}</strong><small>结构化图表</small></span><span><strong>{result.chart_validations?.filter((item) => item.needs_review).length ?? 0}</strong><small>需要复核</small></span><span><strong>{result.cross_modal_checks?.filter((item) => item.status === "supported").length ?? 0}</strong><small>证据互证</small></span></div></div>
      {figures.length === 0 ? <EmptyState icon="chart" title="没有检测到可展示图表" text="这可能表示资料没有图表、图表分支未运行，或检测结果为空。" /> : <div className="figure-grid">{figures.map((figure) => <FigureCard key={figure.figure_id} figure={figure} extraction={extractionByFigure.get(figure.figure_id)} validation={validationByFigure.get(figure.figure_id)} correction={correctionByFigure.get(figure.figure_id)} crossModalCheck={crossModalBySubject.get(figure.figure_id)} />)}</div>}
    </section>
  );
}

export function EvidencePanel({
  result,
  onSelectRecord,
}: {
  result?: AgentResult | null;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  const traces = result?.evidence_traces ?? [];
  const records = useMemo(() => new Map([
    ...(result?.records ?? []),
    ...(result?.dynamic_records ?? []),
  ].map((record) => [record.record_id, record] as const)), [result]);
  const [status, setStatus] = useState("all");
  const filtered = status === "all" ? traces : traces.filter((trace) => trace.locator_status === status);

  if (!result) return <PanelEmptyState title="Evidence is not ready" text="Evidence traces will appear after records are extracted." />;
  return (
    <section className="panel-card evidence-panel">
      <div className="panel-heading wide">
        <div><span className="eyebrow dark">EVIDENCE TRACE</span><h2>Record-to-source evidence</h2><p>Every extracted record is linked to the strongest location currently known.</p></div>
        <div className="source-summary"><span><strong>{traces.length}</strong><small>traces</small></span><span><strong>{traces.filter((trace) => trace.locator_status === "resolved").length}</strong><small>resolved</small></span><span><strong>{traces.filter((trace) => trace.locator_status !== "resolved").length}</strong><small>review</small></span></div>
      </div>
      <div className="table-controls evidence-controls">
        <select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All locator states</option><option value="resolved">Resolved</option><option value="partial">Partial</option><option value="unresolved">Unresolved</option></select>
        <span className="result-count">{filtered.length} traces</span>
      </div>
      {filtered.length === 0 ? <EmptyState icon="link" title="No evidence traces" text="The result has no extracted records with traceable evidence yet." /> : (
        <div className="evidence-trace-list">
          {filtered.map((trace) => <EvidenceTraceCard key={trace.evidence_id} trace={trace} record={records.get(trace.record_id)} onSelectRecord={onSelectRecord} />)}
        </div>
      )}
    </section>
  );
}

function EvidenceTraceCard({
  trace,
  record,
  onSelectRecord,
}: {
  trace: EvidenceTrace;
  record?: EvidenceRecord;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  return (
    <article className="evidence-trace-card">
      <div className="evidence-trace-top"><div><QualityBadge tone={traceTone(trace.locator_status)}>{trace.locator_status}</QualityBadge><span className="evidence-kind">{trace.evidence_type}</span></div><Confidence value={trace.confidence} /></div>
      <div className="evidence-trace-body">
        <div className="evidence-trace-record"><small>RECORD</small><strong>{record ? recordTitle(record) : trace.record_id}</strong></div>
        <div className="evidence-trace-grid">
          <span><small>Source</small><strong>{trace.source_title || trace.source_file}</strong></span>
          <span><small>Location</small><strong>{trace.page ? `Page ${trace.page}` : "Page unknown"}{trace.section_title ? ` / ${trace.section_title}` : ""}</strong></span>
          <span><small>Artifact</small><strong>{trace.table_id || trace.figure_id || trace.artifact_id || "Not resolved"}</strong></span>
          <span><small>Method</small><strong>{trace.extraction_method || "Not recorded"}</strong></span>
        </div>
        {trace.evidence_text && <blockquote>{trace.evidence_text}</blockquote>}
        {trace.notes.length > 0 && <div className="evidence-trace-notes">{trace.notes.slice(0, 2).map((note) => <span key={note}><Icon name="info" size={13} />{note}</span>)}</div>}
      </div>
      {record && <button type="button" className="evidence-open-record" onClick={() => onSelectRecord(record)}><Icon name="link" size={14} />Open record details<Icon name="chevron" size={14} /></button>}
    </article>
  );
}

function FigureCard({ figure, extraction, validation, correction, crossModalCheck }: { figure: FigureAsset; extraction?: ChartExtraction; validation?: ChartValidation; correction?: ChartCorrection; crossModalCheck?: CrossModalCheck }) {
  const points = extraction?.series?.reduce((sum, series) => sum + (series.points?.length ?? 0), 0) ?? 0;
  return (
    <article className="figure-card">
      <div className="figure-image">{figure.image_url ? <ApiAssetImage path={figure.image_url} alt={figure.caption || figure.label || figure.figure_id} /> : <div><Icon name="chart" size={28} /><span>图像文件不可访问</span></div>}<span className="figure-page">第 {figure.page} 页</span></div>
      <div className="figure-body"><div className="figure-title-row"><div><small>{figure.label || figure.figure_id}</small><h3>{extraction?.title || figure.caption || "未命名图表"}</h3></div>{validation?.needs_review ? <QualityBadge tone="warning">需复核</QualityBadge> : validation?.passed ? <QualityBadge tone="success">校验通过</QualityBadge> : <QualityBadge tone="neutral">未校验</QualityBadge>}</div><p className="figure-caption">{figure.caption || "没有提取到 Caption。"}</p><div className="figure-stats"><span><small>类型</small><strong>{extraction?.chart_type || "未分类"}</strong></span><span><small>数据点</small><strong>{points}</strong></span><span><small>VL 置信度</small><strong>{extraction?.confidence !== undefined ? `${Math.round(extraction.confidence * 100)}%` : "—"}</strong></span></div>{extraction && <div className="axis-row"><span>X · {axisLabel(extraction.x_axis)}</span><span>Y · {axisLabel(extraction.y_axis)}</span>{extraction.approximate && <QualityBadge tone="info">近似读数</QualityBadge>}</div>}{correction && <div className={`chart-correction-status chart-correction-${correction.decision}`}><strong>二次复查：{chartCorrectionLabel(correction)}</strong><span>{correction.decision_reason[0] || "已记录初次与复查结果。"}</span></div>}{validation?.issues?.slice(0, 2).map((issue) => <div className="chart-issue" key={issue.code}><Icon name="warning" size={14} />{issue.message}</div>)}{crossModalCheck && <div className={`cross-modal-status cross-modal-${crossModalCheck.status}`}><strong>Cross-modal: {crossModalCheck.status}</strong><span>{crossModalCheck.matched_value_count}/{crossModalCheck.candidate_value_count} numeric matches · {crossModalCheck.modalities.join(" + ")}</span></div>}</div>
    </article>
  );
}

export function QualityPanel({
  task,
  onSelectRecord,
}: {
  task: TaskResponse;
  onSelectRecord: (record: EvidenceRecord) => void;
}) {
  const quality = task.quality_report;
  const result = task.result;
  const coverage = result?.coverage_report;
  const recordById = useMemo(() => new Map([
    ...(result?.records ?? []),
    ...(result?.dynamic_records ?? []),
    ...(result?.needs_review_records ?? []),
  ].map((record) => [record.record_id, record])), [result]);

  if (!quality) return <PanelEmptyState title="质量报告尚未生成" text="任务完成质量校验后，这里会显示覆盖率、警告和跨来源冲突。" />;

  return (
    <div className="quality-layout">
      <div className="quality-hero panel-card">
        <div className="quality-score-ring" style={{ "--score": `${Math.round((quality.provenance_page_coverage ?? 0) * 100) * 3.6}deg` } as React.CSSProperties}><div><strong>{Math.round((quality.provenance_page_coverage ?? 0) * 100)}</strong><small>页码验证</small></div></div>
        <div className="quality-hero-copy"><span className="eyebrow dark">QUALITY REPORT</span><h2>{quality.error_count ? "存在需要处理的质量错误" : quality.warning_count ? "结果可用，但仍需检查警告" : "当前校验未发现显著问题"}</h2><p>质量报告反映现有证据覆盖情况，不代表对科研结论本身的同行评审。</p><div><QualityBadge tone={quality.error_count ? "danger" : "success"}>{quality.error_count ?? 0} 错误</QualityBadge><QualityBadge tone={quality.warning_count ? "warning" : "neutral"}>{quality.warning_count ?? 0} 警告</QualityBadge><QualityBadge tone={quality.conflict_count ? "warning" : "neutral"}>{quality.conflict_count ?? 0} 冲突</QualityBadge></div></div>
        <div className="quality-stat-stack"><span><small>记录</small><strong>{quality.total_record_count ?? quality.record_count ?? 0}</strong></span><span><small>来源</small><strong>{quality.source_count ?? 0}</strong></span><span><small>待复核</small><strong>{Math.max(result?.needs_review_records?.length ?? 0, quality.review_count ?? 0)}</strong></span></div>
      </div>

      {coverage && <CoverageAuditPanel report={coverage} />}
      {result && <AgentRuntimePanel result={result} />}

      <div className="quality-columns">
        <section className="panel-card coverage-panel">
          <div className="panel-heading"><div><h2>覆盖率</h2><p>必填字段与证据的完整程度。</p></div></div>
          <CoverageRow label="证据文本存在率" value={quality.evidence_text_coverage ?? quality.evidence_coverage ?? 0} tone="blue" />
          <CoverageRow label="数值证据覆盖率" value={quality.value_evidence_coverage ?? 0} tone="green" />
          <CoverageRow label="PDF 页码验证率" value={quality.provenance_page_coverage ?? 0} tone="violet" />
          <CoverageRow label="无警告记录率" value={quality.warning_free_rate ?? 0} tone="green" />
          <div className="field-coverage-list">{Object.entries(quality.field_coverage ?? {}).sort((left, right) => left[1] - right[1]).map(([field, value]) => <CoverageRow key={field} label={humanize(field)} value={value} tone={value < 0.6 ? "amber" : "violet"} compact />)}</div>
        </section>

        <section className="panel-card issues-panel">
          <div className="panel-heading"><div><h2>问题与警告</h2><p>点击关联记录可打开证据详情。</p></div><QualityBadge tone={(quality.issues?.length ?? 0) > 0 ? "warning" : "success"}>{quality.issues?.length ?? 0} 项</QualityBadge></div>
          {(quality.issues?.length ?? 0) === 0 ? <EmptyState icon="check" title="没有结构化质量问题" text="仍建议抽查高影响字段与来源证据。" success /> : <div className="issue-list">{quality.issues?.map((issue, index) => {
            const record = issue.record_id ? recordById.get(issue.record_id) : undefined;
            return <button type="button" key={`${issue.record_id}:${issue.field}:${index}`} disabled={!record} onClick={() => record && onSelectRecord(record)}><span className={`issue-icon issue-${issue.level}`}><Icon name={issue.level === "info" ? "info" : "warning"} size={15} /></span><div><span><QualityBadge tone={issue.level === "error" ? "danger" : issue.level === "warning" ? "warning" : "info"}>{issue.level}</QualityBadge>{issue.field && <small>{humanize(issue.field)}</small>}</span><strong>{issue.message}</strong>{issue.record_id && <p>{issue.record_id}</p>}</div>{record && <Icon name="chevron" size={15} />}</button>;
          })}</div>}
        </section>
      </div>

      <section className="panel-card conflicts-panel">
        <div className="panel-heading"><div><span className="eyebrow dark">CROSS-SOURCE CHECK</span><h2>跨来源冲突</h2><p>保留不同来源的值，不静默覆盖。</p></div></div>
        {(quality.conflicts?.length ?? 0) === 0 ? <EmptyState icon="check" title="没有检测到跨来源冲突" text="仅表示当前抽取记录中未命中冲突规则。" success /> : <div className="conflict-grid">{quality.conflicts?.map((conflict) => <article key={conflict.conflict_id}><div><QualityBadge tone="warning">{humanize(conflict.metric_name)}</QualityBadge><small>{conflict.sources.length} 个来源</small></div><h3>{conflict.entity || "未命名实体"}</h3><div className="conflict-values">{conflict.values.map((value, index) => <span key={`${value}:${index}`}>{value}<small>{conflict.sources[index] || "未知来源"}</small></span>)}</div><p>{conflict.message}</p>{conflict.alignment_context && Object.keys(conflict.alignment_context).length > 0 && <small className="conflict-context">Aligned context: {Object.entries(conflict.alignment_context).map(([key, value]) => `${humanize(key)}=${value}`).join("; ")}</small>}{conflict.resolution && <small className="conflict-resolution">Resolution: {conflict.resolution}</small>}</article>)}</div>}
      </section>
    </div>
  );
}

export function CoverageAuditPanel({ report }: { report: CoverageReport }) {
  const allowedToStop = report.decision === "allow_stop";
  const missing = report.missing_requirements ?? [];
  const gaps = report.gaps ?? [];
  return (
    <section className="panel-card coverage-audit-panel">
      <div className="panel-heading wide">
        <div>
          <span className="eyebrow dark">COVERAGE AUDIT</span>
          <h2>任务完成度审核</h2>
          <p>这个审核结果决定 Agent 是否可以停止，不等同于模型自己的 stop 判断。</p>
        </div>
        <QualityBadge tone={allowedToStop ? "success" : "warning"}>
          {allowedToStop ? "允许结束" : "仍需继续"}
        </QualityBadge>
      </div>
      <div className="coverage-audit-summary">
        <span><small>覆盖分数</small><strong>{Math.round((report.coverage_score ?? 0) * 100)}%</strong></span>
        <span><small>字段缺口</small><strong>{missing.length}</strong></span>
        <span><small>未处理高相关资料</small><strong>{report.unprocessed_relevant_artifacts?.length ?? 0}</strong></span>
      </div>
      {missing.length > 0 && <div className="coverage-audit-block"><small>待补齐字段</small><div className="coverage-chip-list">{missing.map((item) => <QualityBadge key={item} tone="warning">{humanize(item)}</QualityBadge>)}</div></div>}
      {gaps.length > 0 && <div className="coverage-gap-block">
        <div className="coverage-gap-heading"><small>EVIDENCE GAPS</small><QualityBadge tone="warning">{gaps.length} gaps</QualityBadge></div>
        <div className="coverage-gap-list">{gaps.map((gap) => <article key={gap.gap_id} className={`coverage-gap coverage-gap-${gap.priority}`}>
          <div className="coverage-gap-title"><strong>{humanize(gap.requirement_name)}</strong><span><QualityBadge tone={gap.priority === "high" ? "danger" : gap.priority === "medium" ? "warning" : "neutral"}>{gap.priority}</QualityBadge><QualityBadge tone={gap.status === "partial" ? "warning" : "danger"}>{gap.status}</QualityBadge></span></div>
          <p>{gap.reason}</p>
          <div className="coverage-gap-meta">
            {gap.missing_fields?.length > 0 && <span>fields: {gap.missing_fields.map(humanize).join(", ")}</span>}
            {gap.missing_evidence_types?.length > 0 && <span>evidence: {gap.missing_evidence_types.join(", ")}</span>}
            <span>evidence count: {gap.evidence_count ?? 0}</span>
          </div>
          {gap.recommended_actions?.length > 0 && <small className="coverage-gap-actions">next: {gap.recommended_actions.join(" / ")}</small>}
        </article>)}</div>
      </div>}
      {report.requirements?.length > 0 && <div className="coverage-requirement-list">{report.requirements.map((item) => <div key={item.name}><span>{humanize(item.name)}</span><QualityBadge tone={item.status === "covered" ? "success" : item.status === "partial" ? "warning" : "danger"}>{item.status} · {item.evidence_count} 条证据</QualityBadge></div>)}</div>}
      {report.reasons?.length > 0 && <div className="coverage-audit-reasons"><small>继续原因</small>{report.reasons.slice(0, 3).map((reason) => <p key={reason}><Icon name="info" size={14} />{reason}</p>)}</div>}
      {report.recommended_actions?.length > 0 && <div className="coverage-audit-actions"><small>建议动作</small><span>{report.recommended_actions.join(" · ")}</span></div>}
    </section>
  );
}

export function AgentRuntimePanel({ result }: { result: AgentResult }) {
  const decisions = result.agent_decision_history ?? [];
  const tools = result.tool_result_history ?? [];
  const trace = result.agent_trace ?? [];
  const failedTools = tools.filter((item) => item.status === "failed");
  const runtimeStatus = result.runtime_status || "legacy_pipeline";
  const stopReason = result.runtime_stop_reason || "No runtime stop reason recorded.";
  return (
    <section className="panel-card agent-runtime-panel">
      <div className="panel-heading wide">
        <div>
          <span className="eyebrow dark">AGENT RUNTIME</span>
          <h2>Decision and tool trace</h2>
        </div>
        <QualityBadge tone={runtimeStatus === "completed" ? "success" : runtimeStatus === "partial" ? "warning" : "info"}>
          {runtimeStatus}
        </QualityBadge>
      </div>
      <div className="agent-runtime-summary">
        <span><small>Iterations</small><strong>{result.runtime_iteration ?? 0}</strong></span>
        <span><small>Decisions</small><strong>{decisions.length}</strong></span>
        <span><small>Tool results</small><strong>{tools.length}</strong></span>
        <span><small>Failures</small><strong>{failedTools.length}</strong></span>
      </div>
      <div className="agent-runtime-stop"><small>Stop reason</small><strong>{stopReason}</strong></div>
      {result.stop_rejections && result.stop_rejections.length > 0 && (
        <div className="agent-runtime-rejections">
          <small>Rejected decisions</small>
          {result.stop_rejections.slice(-3).map((item) => <span key={item}><Icon name="warning" size={13} />{item}</span>)}
        </div>
      )}
      {trace.length > 0 && (
        <div className="agent-runtime-timeline">
          {trace.slice(-10).map((event, index) => (
            <div className="agent-runtime-event" key={`${event.event_id ?? event.event_type}-${index}`}>
              <span className={`runtime-event-dot runtime-event-${event.status || "neutral"}`} />
              <div><strong>{event.event_type.replaceAll("_", " ")}</strong><small>iteration {event.iteration ?? 0}{event.tool_name ? ` / ${event.tool_name}` : ""}{event.status ? ` / ${event.status}` : ""}</small></div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

const exportDescriptions: Record<string, { name: string; description: string; kind: "data" | "quality" | "research" | "log" }> = {
  coverage_report: { name: "任务覆盖审核", description: "动态字段、证据类型和未处理高相关资料", kind: "quality" },
  csv: { name: "科研指标 CSV", description: "清洗后的指标型结构化记录", kind: "data" },
  json: { name: "完整结果 JSON", description: "任务计划、来源、记录与质量报告", kind: "data" },
  dynamic_records: { name: "动态记录", description: "按本次动态 Schema 整理的记录", kind: "data" },
  dynamic_records_clean: { name: "清洗后动态记录", description: "归一、去重后的动态数据", kind: "data" },
  dynamic_records_raw: { name: "原始动态记录", description: "模型抽取后的原始记录快照", kind: "data" },
  dynamic_schema: { name: "动态 Schema", description: "字段、类型、描述与证据要求", kind: "data" },
  needs_review: { name: "复核队列 JSON", description: "证据不足或低置信度记录", kind: "quality" },
  needs_review_csv: { name: "复核队列 CSV", description: "便于人工审核与标注", kind: "quality" },
  quality_report: { name: "质量报告", description: "覆盖率、警告、错误与冲突", kind: "quality" },
  chart_validation: { name: "图表校验报告", description: "坐标轴、序列、单位与复核标记", kind: "quality" },
  cross_modal_validation: { name: "跨模态校验报告", description: "文本、表格与图像证据的互证结果", kind: "quality" },
  chart_extractions: { name: "图表结构化结果", description: "视觉模型读取的坐标轴与数据点", kind: "research" },
  source_catalog: { name: "来源目录", description: "多源材料、Artifact 与处理状态", kind: "research" },
  source_discovery_plan: { name: "来源发现计划", description: "搜索方向、关键词和候选来源", kind: "research" },
  source_selection: { name: "来源选择计划", description: "相关性判断与选择理由", kind: "research" },
  source_triage: { name: "来源分诊记录", description: "下载、解析与跳过决策", kind: "research" },
  paper_survey: { name: "论文调研表", description: "按论文聚合的方法、数据集与指标", kind: "research" },
  connector_status: { name: "连接器状态", description: "数据源查询成功、失败与原因", kind: "log" },
  agent_trace: { name: "Agent Trace", description: "决策、工具生命周期与失败恢复事件", kind: "log" },
  decision_history: { name: "决策历史", description: "LLM 根据观察状态生成的动作决策", kind: "log" },
  tool_history: { name: "工具结果历史", description: "每次工具调用的结果、证据和幂等状态", kind: "log" },
  discovered_sources: { name: "检索结果", description: "各连接器返回的规范化来源", kind: "research" },
  summary: { name: "任务摘要", description: "供演示和前端快速读取的摘要", kind: "research" },
  final_report: { name: "调研报告 Markdown", description: "适合阅读和后续整理的最终报告", kind: "research" },
  processing_log: { name: "处理日志", description: "完整节点与模型处理日志", kind: "log" },
};

export function ExportsPanel({ task }: { task: TaskResponse }) {
  const entries = Object.entries(task.download_urls);
  const reportPath = task.download_urls.final_report;
  const report = useQuery({
    queryKey: ["final-report", task.task_id, reportPath],
    queryFn: () => getTextAsset(reportPath),
    enabled: Boolean(reportPath),
    staleTime: Infinity,
  });
  return (
    <section className="panel-card exports-panel">
      <div className="panel-heading wide"><div><span className="eyebrow dark">DELIVERABLES</span><h2>导出与调研报告</h2><p>所有下载均使用受控 API URL，不暴露服务器文件路径。</p></div><QualityBadge tone={entries.length ? "success" : "neutral"}>{entries.length} 个文件</QualityBadge></div>
      {reportPath && <details className="report-preview"><summary><span><Icon name="document" size={18} /><strong>在线预览调研报告</strong></span><small>{report.isLoading ? "读取中…" : report.isError ? "读取失败" : "展开查看 Markdown"}</small></summary>{report.data && <pre>{report.data}</pre>}{report.isError && <p>报告预览暂时不可用，仍可从下方下载原文件。</p>}</details>}
      {entries.length === 0 ? <EmptyState icon="download" title="导出文件尚未生成" text={task.status === "failed" ? "任务失败前没有完成导出。" : "任务完成 export 节点后，文件会自动出现在这里。"} /> : <div className="export-grid">{entries.map(([format, path]) => {
        const meta = exportDescriptions[format] ?? { name: humanize(format), description: "任务生成的结构化交付文件", kind: "data" as const };
        return <article key={format}><span className={`export-icon export-${meta.kind}`}><Icon name={meta.kind === "quality" ? "shield" : meta.kind === "log" ? "list" : meta.kind === "research" ? "document" : "table"} size={19} /></span><div><small>{format.toUpperCase()}</small><h3>{meta.name}</h3><p>{meta.description}</p></div><ApiAssetLink path={path} download ariaLabel={`下载${meta.name}`}><Icon name="download" size={17} />下载</ApiAssetLink></article>;
      })}</div>}
    </section>
  );
}

export function EvidenceDrawer({
  record,
  taskId,
  reviewDecision,
  sources,
  onClose,
}: {
  record: EvidenceRecord;
  taskId: string;
  reviewDecision?: ReviewDecision;
  sources: SourceCatalogEntry[];
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const drawerRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const [reviewNote, setReviewNote] = useState(reviewDecision?.note ?? "");
  const source = sources.find((item) => item.title.includes(record.source_file) || record.source_file.includes(item.title));
  const sourceAsset = source?.artifacts?.find((artifact) => artifact.asset_url);
  const fields = evidenceFields(record);
  const tableName = evidenceTableName(record);
  const reviewMutation = useMutation({
    mutationFn: (decision: ReviewDecision["decision"]) => submitReview(taskId, record.record_id, decision, reviewNote),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["task", taskId] });
    },
  });

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButtonRef.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !drawerRef.current) return;
      const focusable = Array.from(drawerRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previouslyFocused?.focus();
    };
  }, [onClose]);

  const reviewError = reviewMutation.error instanceof ApiError ? reviewMutation.error.message : null;
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <aside ref={drawerRef} className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="evidence-title">
        <header><div><span className="eyebrow dark">PROVENANCE</span><h2 id="evidence-title">字段与证据详情</h2></div><button ref={closeButtonRef} type="button" onClick={onClose} aria-label="关闭证据详情"><Icon name="close" size={19} /></button></header>
        <div className="drawer-status-row"><QualityBadge tone={record.warnings.length ? "warning" : record.evidence_text ? "success" : "danger"}>{record.warnings.length ? "需要检查" : record.evidence_text ? "已绑定证据" : "缺少证据"}</QualityBadge><span>{record.record_id}</span></div>
        <section className="drawer-section"><small>记录信息</small><div className="evidence-meta-grid"><span><small>数据表</small><strong>{humanize(tableName)}</strong></span><span><small>置信度</small><strong>{Math.round(record.confidence * 100)}%</strong></span><span><small>来源类型</small><strong>{record.source_type || "—"}</strong></span><span><small>页码</small><strong>{record.page ? `第 ${record.page} 页` : "—"}</strong></span></div></section>
        <section className="drawer-section"><small>结构化字段</small><div className="evidence-fields">{Object.entries(fields).map(([key, value]) => <article key={key}><span>{humanize(key)}</span><strong>{displayValue(value)}</strong></article>)}</div></section>
        <section className="drawer-section evidence-source"><small>来源</small><div><span className="section-icon"><Icon name="document" size={17} /></span><div><strong>{record.paper_title || source?.title || record.source_file}</strong><p>{record.source_file}{record.page ? ` · 第 ${record.page} 页` : ""}</p></div>{sourceAsset?.asset_url ? <ApiAssetLink path={sourceAsset.asset_url} ariaLabel="打开证据来源"><Icon name="external" size={15} />打开</ApiAssetLink> : source?.url ? <a href={source.url} target="_blank" rel="noreferrer"><Icon name="external" size={15} />原文</a> : null}</div></section>
        <section className="drawer-section evidence-quote"><small>原文证据</small>{record.evidence_text ? <blockquote>{record.evidence_text}</blockquote> : <div className="missing-evidence"><Icon name="warning" size={17} />该记录没有绑定可展示的原文证据。</div>}</section>
        {record.warnings.length > 0 && <section className="drawer-section"><small>警告</small><div className="drawer-warnings">{record.warnings.map((warning) => <p key={warning}><Icon name="warning" size={15} />{warning}</p>)}</div></section>}
        <section className="drawer-section review-actions">
          <small>人工复核</small>
          {reviewDecision && <p className="review-current"><Icon name="check" size={15} />当前结论：{reviewDecisionLabel(reviewDecision.decision)} · {new Date(reviewDecision.updated_at).toLocaleString("zh-CN")}</p>}
          <label><span className="sr-only">复核备注</span><textarea rows={3} maxLength={2000} value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="可选：填写判断依据或需要修改的内容" /></label>
          <div>
            <button type="button" disabled={reviewMutation.isPending} onClick={() => reviewMutation.mutate("approved")}>确认通过</button>
            <button type="button" disabled={reviewMutation.isPending} onClick={() => reviewMutation.mutate("needs_changes")}>需要修改</button>
            <button type="button" disabled={reviewMutation.isPending} onClick={() => reviewMutation.mutate("rejected")}>拒绝记录</button>
          </div>
          {reviewError && <p className="form-error"><Icon name="warning" size={15} />{reviewError}</p>}
        </section>
        {Object.keys(record.raw ?? {}).length > 0 && <details className="raw-details"><summary>查看原始记录</summary><pre>{JSON.stringify(record.raw, null, 2)}</pre></details>}
      </aside>
    </div>
  );
}

function ApiAssetLink({
  path,
  children,
  download = false,
  ariaLabel,
}: {
  path: string;
  children: ReactNode;
  download?: boolean;
  ariaLabel?: string;
}) {
  const [error, setError] = useState<string | null>(null);
  return (
    <a
      href={apiUrl(path)}
      target={download ? undefined : "_blank"}
      rel={download ? undefined : "noreferrer"}
      download={download || undefined}
      aria-label={ariaLabel}
      aria-invalid={error ? true : undefined}
      title={error ?? undefined}
      onClick={API_AUTH_ENABLED ? (event) => {
        event.preventDefault();
        setError(null);
        void openApiAsset(path, {
          download,
          filename: download ? path.split("/").pop()?.split("?")[0] : undefined,
        }).catch((reason: unknown) => {
          setError(reason instanceof Error ? reason.message : "文件打开失败，请重试。");
        });
      } : undefined}
    >
      {children}
      {error && <span className="sr-only" role="alert">{error}</span>}
    </a>
  );
}

function ApiAssetImage({ path, alt }: { path: string; alt: string }) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!API_AUTH_ENABLED) return undefined;
    let active = true;
    let createdUrl: string | null = null;
    setObjectUrl(null);
    setFailed(false);
    void getAssetBlob(path)
      .then((blob) => {
        if (!active) return;
        createdUrl = URL.createObjectURL(blob);
        setObjectUrl(createdUrl);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [path]);

  const src = API_AUTH_ENABLED ? objectUrl : apiUrl(path);
  if (failed) return <div><Icon name="warning" size={28} /><span>图像文件读取失败</span></div>;
  if (!src) return <div><Icon name="chart" size={28} /><span>图像加载中…</span></div>;
  return <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />;
}

function CoverageRow({ label, value, tone, compact = false }: { label: string; value: number; tone: "blue" | "green" | "violet" | "amber"; compact?: boolean }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return <div className={`coverage-row${compact ? " compact" : ""}`}><div><span>{label}</span><strong>{percent}%</strong></div><div className={`coverage-track coverage-${tone}`}><span style={{ width: `${percent}%` }} /></div></div>;
}

function Confidence({ value }: { value: number }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return <span className={`confidence confidence-${percent >= 80 ? "high" : percent >= 60 ? "medium" : "low"}`}><i><b style={{ width: `${percent}%` }} /></i><strong>{percent}%</strong></span>;
}

function EmptyState({ icon, title, text, success = false }: { icon: "table" | "database" | "chart" | "download" | "check" | "link"; title: string; text: string; success?: boolean }) {
  return <div className={`empty-state${success ? " success" : ""}`}><span><Icon name={icon} size={22} /></span><strong>{title}</strong><p>{text}</p></div>;
}

function PanelEmptyState({ title, text }: { title: string; text: string }) {
  return <div className="panel-card standalone-empty"><EmptyState icon="table" title={title} text={text} /></div>;
}

function humanize(value?: string | null): string {
  if (!value) return "—";
  return value.replaceAll("_", " ");
}

function recordTitle(record: EvidenceRecord): string {
  const entries = Object.entries(evidenceFields(record)).filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (record.paper_title) return record.paper_title;
  if (entries.length === 0) return record.record_id;
  return entries.slice(0, 2).map(([, value]) => displayValue(value)).join(" · ");
}

function recordFieldSummary(record: EvidenceRecord): string {
  const entries = Object.entries(evidenceFields(record)).filter(([, value]) => value !== null && value !== undefined && value !== "");
  return entries.length > 0
    ? entries.slice(0, 2).map(([, value]) => displayValue(value)).join(" · ")
    : evidenceTableName(record);
}

function isDynamicRecord(record: EvidenceRecord): record is DynamicRecord {
  return "fields" in record && "table_name" in record;
}

function evidenceFields(record: EvidenceRecord): Record<string, unknown> {
  if (isDynamicRecord(record)) return record.fields;
  return {
    material: record.material,
    method: record.method,
    metric_name: record.metric_name,
    metric_value: record.metric_value,
    unit: record.unit,
    condition: record.condition,
  };
}

function evidenceTableName(record: EvidenceRecord): string {
  return isDynamicRecord(record) ? record.table_name : "scientific_metrics";
}

function reviewDecisionLabel(decision: ReviewDecision["decision"]): string {
  if (decision === "approved") return "确认通过";
  if (decision === "needs_changes") return "需要修改";
  return "拒绝记录";
}

function chartCorrectionLabel(correction: ChartCorrection): string {
  if (correction.decision === "accepted_second") return "已采用二次结果";
  if (correction.decision === "manual_review") return "两次结果冲突，需人工复核";
  if (correction.decision === "second_pass_failed") return "二次复查失败，保留初次结果";
  return "保留初次结果";
}

function recordsForSource(source: SourceCatalogEntry, records: DynamicRecord[]): DynamicRecord[] {
  const title = source.title.toLowerCase();
  const sourceUrl = source.url?.toLowerCase() ?? "";
  return records.filter((record) => {
    const filename = record.source_file.toLowerCase();
    const stem = filename.replace(/\.[^.]+$/, "");
    const paperTitle = record.paper_title?.toLowerCase() ?? "";
    return paperTitle === title || title.includes(stem) || stem.includes(title) || sourceUrl.includes(filename);
  });
}

function sourceStatusLabel(status?: string | null): string {
  const labels: Record<string, string> = {
    discovered: "已发现",
    selected: "已选择",
    metadata_read: "已读元数据",
    downloaded: "已下载",
    parsed: "已解析",
    failed: "失败",
    skipped: "已跳过",
    planned: "已规划",
    completed: "已完成",
    success: "成功",
  };
  return status ? labels[status] ?? humanize(status) : "未知";
}

function sourceTone(status?: string | null): "success" | "warning" | "danger" | "neutral" | "info" {
  if (status === "parsed" || status === "completed" || status === "success") return "success";
  if (status === "failed") return "danger";
  if (status === "downloaded" || status === "selected") return "info";
  if (status === "skipped") return "warning";
  return "neutral";
}

function traceTone(status: EvidenceTrace["locator_status"]): "success" | "warning" | "danger" | "neutral" | "info" {
  if (status === "resolved") return "success";
  if (status === "partial") return "warning";
  return "danger";
}

function axisLabel(axis?: ChartExtraction["x_axis"]): string {
  if (!axis) return "未识别";
  const label = axis.label || "未命名";
  return axis.unit ? `${label} (${axis.unit})` : label;
}
