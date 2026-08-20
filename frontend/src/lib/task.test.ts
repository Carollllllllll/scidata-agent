import { describe, expect, it } from "vitest";

import { displayValue, formatBytes, progressPercent, stageLabel } from "./task";

describe("task presentation helpers", () => {
  it("maps lifecycle stages to readable Chinese labels", () => {
    expect(stageLabel("quality_validation")).toBe("质量校验");
    expect(stageLabel("custom_stage")).toBe("custom stage");
  });

  it("uses explicit progress before stage-derived progress", () => {
    expect(progressPercent("running", "source_discovery", { current: 3, total: 4 })).toBe(75);
    expect(progressPercent("completed", "source_discovery", null)).toBe(100);
  });

  it("formats empty and structured values without fabricating data", () => {
    expect(displayValue(null)).toBe("—");
    expect(displayValue({ metric: "PCE" })).toContain("PCE");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });
});
