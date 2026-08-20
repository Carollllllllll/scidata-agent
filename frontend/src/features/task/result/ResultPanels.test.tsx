import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { AgentResult, DynamicRecord } from "../../../types/api";
import { DynamicDataPanel, SourcesPanel } from "./ResultPanels";

function dynamicRecord(overrides: Partial<DynamicRecord> = {}): DynamicRecord {
  return {
    record_id: "dyn_test",
    table_name: "mixed_results",
    fields: {
      field_a: "A",
      field_b: "B",
      field_c: "C",
      field_d: "D",
      field_e: "E",
    },
    source_file: "paper.pdf",
    source_type: "pdf_text",
    page: 3,
    evidence_text: "Evidence on page three.",
    confidence: 0.9,
    warnings: [],
    raw: {},
    ...overrides,
  };
}

describe("result panels", () => {
  it("limits the cross-table overview to four high-frequency fields", () => {
    const result = {
      dynamic_records: [dynamicRecord()],
      dynamic_records_raw: [],
      dynamic_extraction_plan: { dynamic_tables: [] },
    } as unknown as AgentResult;

    const html = renderToStaticMarkup(
      <DynamicDataPanel result={result} onSelectRecord={() => undefined} />,
    );

    expect(html).toContain("field a");
    expect(html).toContain("field d");
    expect(html).not.toContain("field e");
    expect(html).toContain("tabindex=\"0\"");
  });

  it("shows separate source lifecycle counters", () => {
    const result = {
      dynamic_records: [],
      source_catalog: [
        {
          source_id: "source_1",
          title: "Paper one",
          status: "parsed",
          selection_action: "select",
          artifacts: [{ artifact_id: "artifact_1", status: "parsed" }],
        },
      ],
    } as unknown as AgentResult;

    const html = renderToStaticMarkup(
      <SourcesPanel result={result} onSelectRecord={() => undefined} />,
    );

    expect(html).toContain("已发现");
    expect(html).toContain("已选择");
    expect(html).toContain("已下载");
    expect(html).toContain("已解析");
  });
});
