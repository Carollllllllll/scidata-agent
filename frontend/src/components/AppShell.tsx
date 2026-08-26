import { useQuery } from "@tanstack/react-query";
import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { getHealth, listTasks } from "../api/client";
import {
  chooseTaskDetailId,
  forgetRecentTaskId,
  readRecentTaskId,
  rememberTaskId,
  taskIdFromPath,
} from "../lib/recentTask";
import { Icon } from "./Icon";

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 30_000 });
  const tasks = useQuery({ queryKey: ["tasks", 12], queryFn: () => listTasks(12) });
  const inTask = location.pathname.startsWith("/tasks/");
  const inHistory = location.pathname === "/history";
  const currentTaskId = taskIdFromPath(location.pathname);
  const [rememberedTaskId, setRememberedTaskId] = useState(readRecentTaskId);
  const taskDetailId = chooseTaskDetailId(currentTaskId, tasks.data?.tasks, rememberedTaskId);

  useEffect(() => {
    if (!taskDetailId || taskDetailId === rememberedTaskId) return;
    rememberTaskId(taskDetailId);
    setRememberedTaskId(taskDetailId);
  }, [rememberedTaskId, taskDetailId]);

  useEffect(() => {
    if (currentTaskId || !tasks.isSuccess || tasks.data.count !== 0 || !rememberedTaskId) return;
    forgetRecentTaskId();
    setRememberedTaskId(null);
  }, [currentTaskId, rememberedTaskId, tasks.data, tasks.isSuccess]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/" className="brand" aria-label="SciData Agent 首页">
          <img className="brand-mark" src="/scidata-mark.svg" alt="" />
          <span>
            <strong>SciData</strong>
            <small>Agent Workbench</small>
          </span>
        </NavLink>

        <nav className="sidebar-nav" aria-label="主导航">
          <NavLink to="/" end className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <Icon name="home" />
            <span>科研工作台</span>
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <Icon name="clock" />
            <span>历史任务</span>
          </NavLink>
          {taskDetailId ? (
            <NavLink
              to={`/tasks/${taskDetailId}`}
              className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
            >
              <Icon name="layers" />
              <span>任务详情</span>
            </NavLink>
          ) : (
            <span className="nav-item nav-disabled" aria-disabled="true" title="创建任务后可在这里返回任务详情">
              <Icon name="layers" />
              <span>任务详情</span>
            </span>
          )}
        </nav>

        <div className="service-card">
          <div className="service-row">
            <span className={`service-indicator ${health.isSuccess ? "online" : health.isError ? "offline" : "checking"}`} />
            <div>
              <strong>{health.isSuccess ? "API 已连接" : health.isError ? "API 未连接" : "正在检查 API"}</strong>
              <small>{health.data?.version ? `API v${health.data.version}` : "localhost:8000"}</small>
            </div>
          </div>
          <div className="service-row service-agent-row" title={health.data?.model}>
            <span
              className={`service-indicator ${
                health.data?.qwen_configured
                  ? "online"
                  : health.isError
                    ? "offline"
                    : health.isSuccess
                      ? ""
                      : "checking"
              }`}
            />
            <div>
              <strong>
                {health.data?.qwen_configured
                  ? "Agent 就绪"
                  : health.isError
                    ? "Agent 不可用"
                    : health.isSuccess
                      ? "Agent 未配置"
                      : "正在检查 Agent"}
              </strong>
              <small>
                {health.data?.qwen_configured
                  ? "Qwen 已配置"
                  : health.isError
                    ? "请先连接后端"
                    : health.isSuccess
                      ? "等待配置 Qwen"
                      : "读取模型配置"}
              </small>
            </div>
          </div>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="mobile-brand"><img src="/scidata-mark.svg" alt="" /> SciData</div>
          <div className="topbar-breadcrumb">
            <span>科研数据工作台</span>
            {(inTask || inHistory) && (
              <><Icon name="chevron" size={14} /><strong>{inTask ? "任务详情" : "历史任务"}</strong></>
            )}
          </div>
        </header>
        <div className="page-container">{children}</div>
      </main>
    </div>
  );
}
