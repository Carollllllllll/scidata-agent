import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { AgentResult, DynamicRecord, SourceCatalogEntry, TaskResponse } from "../../../types/api";
import { AgentRuntimePanel, DynamicDataPanel, QualityPanel, SourceDetail, SourcesPanel } from "./ResultPanels";

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

  it("shows source lifecycle data", () => {
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

    expect(html).toContain("Paper one");
    expect(html).toContain("未知来源");
    expect(html).toContain("parsed");
  });

  it("shows coverage gaps when the auditor requires another iteration", () => {
    const task = {
      status: "completed",
      uploads: [],
      download_urls: {},
      review_decisions: {},
      quality_report: {},
      result: {
        coverage_report: {
          decision: "continue",
          coverage_score: 0.4,
          gaps: [{
            gap_id: "requirement_model_architecture",
            requirement_name: "model architecture",
            priority: "high",
            status: "missing",
            missing_fields: ["model architecture"],
            missing_evidence_types: [],
            evidence_count: 0,
            reason: "No sufficient field-level evidence has been extracted.",
            recommended_actions: ["parse_pdf_sections"],
          }],
          requirements: [],
          missing_requirements: ["model architecture"],
          required_evidence_types: ["experimental_result"],
          covered_evidence_types: [],
          unprocessed_relevant_artifacts: ["artifact_1"],
          reasons: ["The high-relevance artifact has not been parsed."],
          recommended_actions: ["parse_pdf_sections"],
        },
      },
    } as unknown as TaskResponse;

    const html = renderToStaticMarkup(
      <QualityPanel task={task} onSelectRecord={() => undefined} />,
    );

    expect(html).toContain("COVERAGE AUDIT");
    expect(html).toContain("model architecture");
    expect(html).toContain("未处理高相关资料");
  });

  it("renders artifact relevance and evidence details in source detail", () => {
    const source: SourceCatalogEntry = {
      source_id: "source_1",
      title: "Paper one",
      status: "parsed",
      provider: "arXiv",
      artifacts: [
        {
          artifact_id: "artifact_1",
          name: "paper.pdf",
          artifact_type: "pdf",
          status: "parsed",
          relevance_score: 3.6,
          evidence_types: ["method", "experimental_result"],
          relevance_reason: "Contains the requested architecture and benchmark results.",
        },
      ],
    };

    const html = renderToStaticMarkup(
      <SourceDetail source={source} records={[]} onSelectRecord={() => undefined} />,
    );

    expect(html).toContain("3.6/4");
    expect(html).toContain("method");
    expect(html).toContain("experimental_result");
    expect(html).toContain("Contains the requested architecture and benchmark results.");
  });

  it("renders the Agent decision and tool trace", () => {
    const result = {
      runtime_iteration: 3,
      runtime_status: "partial",
      runtime_stop_reason: "Agent runtime safety budget exhausted after 3 iteration(s).",
      agent_decision_history: [{ decision: "continue", reason: "Collect missing table evidence.", tool_calls: [{ call_id: "call_1", tool_name: "parse_table" }] }],
      tool_result_history: [{ call_id: "call_1", tool_name: "parse_table", status: "failed", errors: ["TATR weights unavailable"] }],
      stop_rejections: ["Coverage gate is 'continue'; required evidence is not complete."],
      agent_trace: [{ event_type: "tool_failed", iteration: 2, tool_name: "parse_table", status: "failed" }],
    } as unknown as AgentResult;

    const html = renderToStaticMarkup(<AgentRuntimePanel result={result} />);

    expect(html).toContain("AGENT RUNTIME");
    expect(html).toContain("safety budget exhausted");
    expect(html).toContain("tool failed");
    expect(html).toContain("parse_table");
    expect(html).toContain("Coverage gate");
  });
});
