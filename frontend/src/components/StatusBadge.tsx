import type { TaskStatus } from "../types/api";
import { STATUS_LABELS } from "../lib/task";

export function StatusBadge({ status, compact = false }: { status: TaskStatus; compact?: boolean }) {
  return (
    <span className={`status-badge status-${status}${compact ? " status-compact" : ""}`}>
      <span className="status-dot" />
      {STATUS_LABELS[status]}
    </span>
  );
}

export function QualityBadge({
  tone,
  children,
}: {
  tone: "success" | "warning" | "danger" | "neutral" | "info";
  children: React.ReactNode;
}) {
  return <span className={`quality-badge quality-${tone}`}>{children}</span>;
}
