import { describe, expect, it } from "vitest";

import type { TaskResponse } from "../types/api";
import {
  chooseTaskDetailId,
  forgetRecentTaskId,
  readRecentTaskId,
  rememberTaskId,
  taskIdFromPath,
} from "./recentTask";

function task(taskId: string, status: TaskResponse["status"]): TaskResponse {
  return {
    task_id: taskId,
    status,
    uploads: [],
    download_urls: {},
    review_decisions: {},
  };
}

function memoryStorage() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    removeItem: (key: string) => { values.delete(key); },
  };
}

describe("recent task navigation", () => {
  it("keeps the current task available after returning to the workbench", () => {
    const current = "20260821_001301_058_e480";
    expect(taskIdFromPath(`/tasks/${current}`)).toBe(current);
    expect(chooseTaskDetailId(null, undefined, current)).toBe(current);
  });

  it("prefers an active task and otherwise uses the latest valid task", () => {
    const tasks = [task("completed_task", "completed"), task("running_task", "running")];
    expect(chooseTaskDetailId(null, tasks, "completed_task")).toBe("running_task");
    expect(chooseTaskDetailId(null, [tasks[0]], null)).toBe("completed_task");
  });

  it("persists only valid task identifiers", () => {
    const storage = memoryStorage();
    rememberTaskId("valid_task_123", storage);
    expect(readRecentTaskId(storage)).toBe("valid_task_123");
    rememberTaskId("../invalid", storage);
    expect(readRecentTaskId(storage)).toBe("valid_task_123");
    forgetRecentTaskId(storage);
    expect(readRecentTaskId(storage)).toBeNull();
  });
});
