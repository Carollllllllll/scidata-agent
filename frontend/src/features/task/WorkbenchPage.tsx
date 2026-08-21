import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";

import { listTasks } from "../../api/client";
import { Icon } from "../../components/Icon";
import { StatusBadge } from "../../components/StatusBadge";
import { formatDate, stageLabel } from "../../lib/task";
import { TaskComposer } from "./TaskComposer";

export function WorkbenchPage() {
  const tasks = useQuery({
    queryKey: ["tasks", 12],
    queryFn: () => listTasks(12),
    refetchInterval: (query) => {
      const hasActiveTask = query.state.data?.tasks.some((task) => task.status === "queued" || task.status === "running");
      if (!hasActiveTask) return false;
      return typeof document !== "undefined" && document.hidden ? false : 4_000;
    },
    refetchOnWindowFocus: true,
  });

  return (
    <div className="workbench-page">
      <section className="page-heading">
        <div>
          <div className="heading-kicker"><span /> SCIENTIFIC DATA AGENT</div>
          <h1>科研数据整合工作台</h1>
          <p>把论文、数据库、表格与图表组织成可追溯、可复核、可导出的结构化科研数据。</p>
        </div>
      </section>

      <TaskComposer />

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
