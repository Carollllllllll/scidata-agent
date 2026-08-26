import { TaskComposer } from "./TaskComposer";

export function WorkbenchPage() {
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
    </div>
  );
}
