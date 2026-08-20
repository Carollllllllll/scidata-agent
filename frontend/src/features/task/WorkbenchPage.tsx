import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { getHealth, listTasks } from "../../api/client";
import { Icon } from "../../components/Icon";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDate, stageLabel } from "../../lib/task";
import { TaskComposer } from "./TaskComposer";

export function WorkbenchPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const tasks = useQuery({ queryKey: ["tasks", 12], queryFn: () => listTasks(12) });

  return (
    <div className="workbench-page">
      <section className="page-heading">
        <div>
          <div className="heading-kicker"><span /> SCIENTIFIC DATA AGENT</div>
          <h1>科研数据整合工作台</h1>
          <p>把论文、数据库、表格与图表组织成可追溯、可复核、可导出的结构化科研数据。</p>
        </div>
        <div className="heading-status">
          <span className={`service-indicator ${health.isSuccess ? "online" : health.isError ? "offline" : "checking"}`} />
          <div><strong>{health.data?.qwen_configured ? "Agent 就绪" : health.isSuccess ? "API 已连接" : health.isError ? "API 未连接" : "检查服务中"}</strong><small>{health.data?.qwen_configured ? "Qwen 已配置" : "在线模型尚未配置"}</small></div>
        </div>
      </section>

      <section className="home-grid">
        <TaskComposer />
        <aside className="workflow-card">
          <div className="workflow-card-heading">
            <span className="section-icon"><Icon name="layers" size={19} /></span>
            <div><strong>Agent 工作流</strong><small>每一步都有状态与证据</small></div>
          </div>
          <ol className="workflow-list">
            <WorkflowItem index="01" title="理解问题" text="生成本次任务专属的字段与质量规则" />
            <WorkflowItem index="02" title="发现来源" text="检索论文、开放数据库、附件与图表" />
            <WorkflowItem index="03" title="解析与对齐" text="处理正文、表格、图片，合并多源字段" />
            <WorkflowItem index="04" title="核验证据" text="标注页码、原文、置信度、冲突与警告" />
            <WorkflowItem index="05" title="结构化交付" text="生成 CSV、JSON、质量报告和调研报告" />
          </ol>
          <div className="truth-note">
            <Icon name="shield" size={18} />
            <span><strong>结果不补写空缺事实</strong><small>未找到、低置信度和来源失败会被明确展示。</small></span>
          </div>
        </aside>
      </section>

      <section className="recent-section">
        <div className="section-heading-row">
          <div><span className="eyebrow dark"><Icon name="clock" size={14} /> TASK HISTORY</span><h2>最近任务</h2></div>
          <button className="icon-button" type="button" onClick={() => tasks.refetch()} disabled={tasks.isFetching} aria-label="刷新任务">
            <Icon name="refresh" size={17} className={tasks.isFetching ? "spin" : ""} />
          </button>
        </div>

        {tasks.isLoading && <div className="task-list-skeleton"><span /><span /><span /></div>}
        {tasks.isError && (
          <div className="empty-panel compact"><Icon name="warning" /><div><strong>无法读取任务历史</strong><p>请先启动后端服务，然后重试。</p></div><button onClick={() => tasks.refetch()}>重试</button></div>
        )}
        {tasks.data?.count === 0 && (
          <div className="empty-panel"><Icon name="folder" size={24} /><div><strong>还没有调研任务</strong><p>从上方输入一个科研问题，首个任务会出现在这里。</p></div></div>
        )}
        {tasks.data && tasks.data.count > 0 && (
          <div className="recent-table-wrap">
            <table className="recent-table">
              <thead><tr><th>研究问题</th><th>当前阶段</th><th>结果规模</th><th>更新时间</th><th>状态</th><th /></tr></thead>
              <tbody>
                {tasks.data.tasks.map((task) => (
                  <tr key={task.task_id}>
                    <td><Link to={`/tasks/${task.task_id}`} className="task-title">{task.research_question || "未命名任务"}<small>{task.task_id}</small></Link></td>
                    <td><span className="stage-cell"><Icon name="layers" size={15} /> {stageLabel(task.current_step)}</span></td>
                    <td>{task.summary?.dynamic_records_extracted ?? task.summary?.records_extracted ?? "—"} 条记录</td>
                    <td>{formatDate(task.updated_at)}</td>
                    <td><StatusBadge status={task.status} compact /></td>
                    <td><Link className="row-action" to={`/tasks/${task.task_id}`} aria-label="查看任务"><Icon name="arrow" size={16} /></Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function WorkflowItem({ index, title, text }: { index: string; title: string; text: string }) {
  return <li><span>{index}</span><div><strong>{title}</strong><small>{text}</small></div></li>;
}
