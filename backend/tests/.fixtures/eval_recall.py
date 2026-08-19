"""Evaluate PDF table extraction recall/precision on a controlled fixture.

Generates a PDF with 4 known tables (3 ruled + 1 whitespace-only) and reports
how many are detected by the active extraction backend.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make scidata_agent importable when running this script directly.
backend_root = Path(__file__).resolve().parents[2]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle

# Force TATR on by default for this evaluation.
os.environ.setdefault("USE_TABLE_TRANSFORMER", "true")

from scidata_agent.agent.schemas import UploadedFile  # noqa: E402
from scidata_agent.tools.parser import parse_pdf_tables  # noqa: E402


def _build_evaluation_pdf(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=letter)
    elements: list = []

    def add_table(title: str, data: list[list[str]], use_grid: bool = True) -> None:
        elements.append(Paragraph(f"<b>{title}</b>", style=None))
        t = RLTable(data, colWidths=[1.4 * inch] * len(data[0]))
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if use_grid:
            style.append(("GRID", (0, 0), (-1, -1), 1, colors.black))
        else:
            style.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey))
        t.setStyle(TableStyle(style))
        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

    # Table 1: ruled, 4 cols x 3 rows
    add_table("Table 1. Device performance.", [
        ["Material", "Method", "PCE (%)", "Condition"],
        ["MAPbI3", "spin coating", "21.3", "AM 1.5G"],
        ["FAPbI3", "annealing", "23.1", "N2"],
    ])

    # Table 2: ruled, 3 cols x 4 rows
    add_table("Table 2. Stability summary.", [
        ["Sample", "Time (h)", "Retention (%)"],
        ["A", "100", "95"],
        ["B", "500", "92"],
        ["C", "1000", "88"],
        ["D", "2000", "80"],
    ])

    # Table 3: whitespace-only (no grid), 3 cols x 2 rows
    add_table("Table 3. Whitespace-only benchmark.", [
        ["Model", "RMSE", "MAE"],
        ["Baseline", "0.15", "0.12"],
        ["Proposed", "0.08", "0.06"],
    ], use_grid=False)

    # Table 4: ruled, 5 cols x 2 rows
    add_table("Table 4. Hyperparameters.", [
        ["Param", "Value", "Unit", "Range", "Note"],
        ["lr", "1e-3", "None", "1e-4 to 1e-2", "Adam"],
        ["batch", "32", "None", "16 to 128", "fixed"],
    ])

    doc.build(elements)


def _ground_truth_headers() -> set[frozenset[str]]:
    return {
        frozenset(["Material", "Method", "PCE (%)", "Condition"]),
        frozenset(["Sample", "Time (h)", "Retention (%)"]),
        frozenset(["Model", "RMSE", "MAE"]),
        frozenset(["Param", "Value", "Unit", "Range", "Note"]),
    }


def _match_detected_to_ground_truth(tables: list) -> tuple[int, int]:
    gt_sets = _ground_truth_headers()
    matched = set()
    false_positives = 0
    for table in tables:
        headers = frozenset(str(c).strip() for c in table.columns)
        if headers in gt_sets and headers not in matched:
            matched.add(headers)
        else:
            false_positives += 1
    return len(matched), false_positives


def main() -> None:
    pdf = Path("tests/.fixtures/evaluation_tables.pdf")
    _build_evaluation_pdf(pdf)

    uploaded = UploadedFile(filename=pdf.name, path=pdf)
    tables = parse_pdf_tables(uploaded, max_pages=None)

    matched, fp = _match_detected_to_ground_truth(tables)
    total_gt = len(_ground_truth_headers())
    recall = matched / total_gt if total_gt else 0.0
    precision = matched / (matched + fp) if (matched + fp) else 0.0

    print(f"Detected {len(tables)} tables (expected {total_gt})")
    for i, t in enumerate(tables):
        print(f"[{i}] p{t.page} {t.extraction_method} cols={len(t.columns)} rows={len(t.rows)} caption={(t.caption or '')[:80]}")
        print(f"    columns: {t.columns}")
    print(f"\nRecall    = {matched}/{total_gt} = {recall:.0%}")
    print(f"Precision = {matched}/{matched + fp} = {precision:.0%}")


if __name__ == "__main__":
    main()
