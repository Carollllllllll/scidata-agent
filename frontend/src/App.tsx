import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import { TaskWorkspace } from "./features/task/TaskWorkspace";
import { WorkbenchPage } from "./features/task/WorkbenchPage";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<WorkbenchPage />} />
        <Route path="/tasks/:taskId" element={<TaskWorkspace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
