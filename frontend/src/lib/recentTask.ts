import type { TaskResponse } from "../types/api";

const RECENT_TASK_STORAGE_KEY = "scidata-agent:recent-task-id";
const TASK_ID_PATTERN = /^[A-Za-z0-9_]+$/;

type TaskStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function isTaskId(value: unknown): value is string {
  return typeof value === "string" && TASK_ID_PATTERN.test(value);
}

export function taskIdFromPath(pathname: string): string | null {
  const match = pathname.match(/^\/tasks\/([^/]+)\/?$/);
  if (!match) return null;
  try {
    const taskId = decodeURIComponent(match[1]);
    return isTaskId(taskId) ? taskId : null;
  } catch {
    return null;
  }
}

export function chooseTaskDetailId(
  currentTaskId: string | null,
  tasks: TaskResponse[] | undefined,
  rememberedTaskId: string | null,
): string | null {
  if (isTaskId(currentTaskId)) return currentTaskId;

  if (tasks) {
    const activeTask = tasks.find(
      (task) => isTaskId(task.task_id) && (task.status === "queued" || task.status === "running"),
    );
    if (activeTask) return activeTask.task_id;

    if (isTaskId(rememberedTaskId) && tasks.some((task) => task.task_id === rememberedTaskId)) {
      return rememberedTaskId;
    }

    const latestTask = tasks.find((task) => isTaskId(task.task_id));
    if (latestTask) return latestTask.task_id;
  }

  return isTaskId(rememberedTaskId) ? rememberedTaskId : null;
}

export function readRecentTaskId(storage: TaskStorage | null = browserStorage()): string | null {
  if (!storage) return null;
  try {
    const value = storage.getItem(RECENT_TASK_STORAGE_KEY);
    return isTaskId(value) ? value : null;
  } catch {
    return null;
  }
}

export function rememberTaskId(taskId: string, storage: TaskStorage | null = browserStorage()): void {
  if (!storage || !isTaskId(taskId)) return;
  try {
    storage.setItem(RECENT_TASK_STORAGE_KEY, taskId);
  } catch {
    // Navigation still works for the current session when browser storage is unavailable.
  }
}

export function forgetRecentTaskId(storage: TaskStorage | null = browserStorage()): void {
  if (!storage) return;
  try {
    storage.removeItem(RECENT_TASK_STORAGE_KEY);
  } catch {
    // Ignore browser storage restrictions.
  }
}

function browserStorage(): TaskStorage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
