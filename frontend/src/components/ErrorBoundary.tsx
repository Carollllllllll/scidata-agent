import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  failed: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("SciData workbench render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.failed) return this.props.children;

    return (
      <main className="fatal-error" role="alert">
        <div className="fatal-error-card">
          <span className="eyebrow">界面恢复</span>
          <h1>页面暂时无法显示</h1>
          <p>某个结果面板渲染失败，任务数据仍保存在后端。刷新后可以继续查看。</p>
          <button type="button" className="primary-button" onClick={() => window.location.reload()}>
            刷新页面
          </button>
        </div>
      </main>
    );
  }
}
