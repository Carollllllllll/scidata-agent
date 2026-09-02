from __future__ import annotations

import csv

from scidata_agent.agent.schemas import AgentState
from scidata_agent.tools.exporter import _write_csv, build_human_summary


def test_csv_export_escapes_spreadsheet_formulas_but_keeps_numbers(tmp_path) -> None:
    path = tmp_path / "safe.csv"
    _write_csv(
        [
            {
                "formula": "=HYPERLINK(\"https://example.invalid\")",
                "plus": "+SUM(1,2)",
                "minus_text": "-2",
                "at": "@cmd",
                "number": -2,
                "normal": "paper title",
            }
        ],
        path,
    )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["formula"].startswith("'=")
    assert row["plus"].startswith("'+")
    assert row["minus_text"] == "'-2"
    assert row["at"] == "'@cmd"
    assert row["number"] == "-2"
    assert row["normal"] == "paper title"


def test_human_summary_preserves_partial_runtime_status(tmp_path) -> None:
    state = AgentState(
        task_id="partial-summary",
        research_question="q",
        files=[],
        output_dir=tmp_path,
        runtime_status="partial",
        runtime_iteration=3,
        runtime_stop_reason="Agent runtime safety budget exhausted after 3 iteration(s).",
    )

    summary = build_human_summary(state, [])

    assert summary["status"] == "partial"
    assert summary["runtime_status"] == "partial"
    assert summary["runtime_iteration"] == 3
