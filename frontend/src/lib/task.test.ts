import { describe, expect, it } from "vitest";

import { displayValue, formatBytes, overallProgressPercent, progressPercent, stageLabel } from "./task";

describe("task presentation helpers", () => {
  it("maps lifecycle stages to readable Chinese labels", () => {
    expect(stageLabel("quality_validation")).toBe("质量校验");
    expect(stageLabel("custom_stage")).toBe("custom stage");
  });

  it("maps stage-local progress into the overall pipeline", () => {
    expect(progressPercent("running", "figure_chart_extraction", { current: 1, total: 12 })).toBe(65);
    expect(progressPercent("running", "figure_chart_extraction", { current: 12, total: 12 })).toBe(70);
    expect(progressPercent("running", "source_parsing", null, "completed")).toBe(64);
    expect(progressPercent("completed", "source_discovery", null)).toBe(100);
  });

  it("returns to the active stage when a dynamic workflow reopens source work", () => {
    const events = [
      { event_type: "step", step: "artifact_action_execution", status: "completed" },
      { event_type: "step", step: "artifact_search_more_source_selection", status: "started" },
      { event_type: "progress", step: "arxiv_pdf_ingestion", status: "started" },
    ];
    expect(overallProgressPercent("running", events, "arxiv_pdf_ingestion", { current: 1, total: 2 })).toBe(41);
    expect(stageLabel("artifact_search_more_source_selection")).toBe("扩展检索：筛选来源");
  });

  it("shows scientific coverage instead of 100% for a partial result", () => {
    expect(overallProgressPercent("partial", [], "partial", null, null, 0.3667)).toBe(37);
    expect(progressPercent("partial", "multi_source_search", null, "started")).toBe(20);
    expect(stageLabel("partial")).toBe("部分完成");
  });

  it("formats empty and structured values without fabricating data", () => {
    expect(displayValue(null)).toBe("—");
    expect(displayValue({ metric: "PCE" })).toContain("PCE");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });
});
