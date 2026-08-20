import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { getHealth } from "../api/client";
import { Icon } from "./Icon";

export function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation();
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 30_000 });
  const inTask = location.pathname.startsWith("/tasks/");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/" className="brand" aria-label="SciData Agent 首页">
          <span className="brand-mark"><Icon name="beaker" size={22} /></span>
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
          {inTask && (
            <span className="nav-item active nav-current">
              <Icon name="layers" />
              <span>任务详情</span>
            </span>
          )}
        </nav>

        <div className="sidebar-context">
          <p>赛道 2 · 方向 1 · A</p>
          <strong>科学数据查找、解析与整合</strong>
          <span>Qwen + 多源证据链</span>
        </div>

        <div className="service-card">
          <div className="service-row">
            <span className={`service-indicator ${health.isSuccess ? "online" : health.isError ? "offline" : "checking"}`} />
            <div>
              <strong>{health.isSuccess ? "服务已连接" : health.isError ? "后端未连接" : "正在检查"}</strong>
              <small>{health.data?.version ? `API v${health.data.version}` : "localhost:8000"}</small>
            </div>
          </div>
          {health.data && (
            <div className="model-chip" title={health.data.model}>
              <Icon name="spark" size={14} />
              {health.data.qwen_configured ? health.data.model : "等待配置 Qwen"}
            </div>
          )}
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="mobile-brand"><Icon name="beaker" size={20} /> SciData</div>
          <div className="topbar-breadcrumb">
            <span>科研数据工作台</span>
            {inTask && <><Icon name="chevron" size={14} /><strong>任务详情</strong></>}
          </div>
          <div className="topbar-meta">
            <span className="privacy-note"><Icon name="shield" size={15} /> 密钥仅由后端读取</span>
            <span className="date-chip">2026 数据场景赛道</span>
          </div>
        </header>
        <div className="page-container">{children}</div>
      </main>
    </div>
  );
}
