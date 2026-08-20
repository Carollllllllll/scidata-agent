from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph
from reportlab.pdfgen import canvas

from scidata_agent.agent.schemas import SourceType, TableBlock, UploadedFile
from scidata_agent.tools.parser import (
    _detect_header_end,
    _table_quality_acceptable,
    parse_pdf_tables,
    parse_sources,
)


def _make_table_pdf(path: Path, title: str = "Demo paper") -> Path:
    """Create a PDF with one page of text and one page containing a real table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    elements = []

    elements.append(Paragraph(f"<b>{title}</b>", style=None))
    elements.append(Paragraph("Table 1. Device performance metrics.", style=None))

    data = [
        ["Material", "Method", "PCE (%)", "Condition"],
        ["MAPbI3", "spin coating", "21.3", "AM 1.5G"],
        ["FAPbI3", "annealing", "23.1", "1000 h N2"],
        ["CsPbI3", "spray", "19.8", "ambient"],
    ]
    table = RLTable(data, colWidths=[1.5 * inch, 1.5 * inch, 1.2 * inch, 1.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    return path


def _make_text_only_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(72, 740, "No tables here")
    pdf.save()
    return path


def test_parse_pdf_tables_extracts_structured_table() -> None:
    pdf_path = Path("tests/.fixtures/with_table.pdf")
    _make_table_pdf(pdf_path)
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    tables = parse_pdf_tables(uploaded, max_pages=None)

    assert len(tables) >= 1, f"expected at least one table, got {len(tables)}"
    table = tables[0]
    assert table.source_type == SourceType.PDF_TABLE
    assert table.page == 1
    assert table.columns == ["Material", "Method", "PCE (%)", "Condition"]
    assert len(table.rows) == 3
    assert table.rows[0]["Material"] == "MAPbI3"
    assert table.rows[0]["PCE (%)"] == 21.3
    assert table.caption and "Table 1" in table.caption
    assert table.bbox is not None and len(table.bbox) == 4


def test_parse_pdf_tables_returns_empty_for_text_only_pdf() -> None:
    pdf_path = Path("tests/.fixtures/text_only.pdf")
    _make_text_only_pdf(pdf_path)
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    tables = parse_pdf_tables(uploaded, max_pages=None)
    assert tables == []


def test_parse_sources_includes_pdf_tables() -> None:
    pdf_path = Path("tests/.fixtures/with_table.pdf")
    _make_table_pdf(pdf_path)
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    parsed = parse_sources([uploaded], max_pdf_pages=None)

    assert any(block.source_type == SourceType.PDF_TEXT for block in parsed.text_blocks)
    assert any(table.source_type == SourceType.PDF_TABLE for table in parsed.tables)
    table = next(table for table in parsed.tables if table.source_type == SourceType.PDF_TABLE)
    assert table.columns[0] == "Material"


def test_pdf_table_quality_filter_rejects_fragment() -> None:
    """A single-line or single-column accidental detection should be filtered out."""
    pdf_path = Path("tests/.fixtures/with_table.pdf")
    _make_table_pdf(pdf_path)
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    tables = parse_pdf_tables(uploaded, max_pages=None)
    for table in tables:
        assert len(table.columns) >= 2 or len(table.rows) >= 3
        # Empty cell ratio must be reasonable.
        total = len(table.columns) * len(table.rows)
        empty = sum(
            1 for row in table.rows for value in row.values() if value is None or str(value).strip() == ""
        )
        assert empty / total < 0.85


def test_text_strategy_rejects_uncaptioned_page_layout() -> None:
    block = TableBlock(
        source_file="paper.pdf",
        source_path="paper.pdf",
        source_type=SourceType.PDF_TABLE,
        columns=["Introduction", "Related work", "Method"],
        rows=[
            {"Introduction": "This is ordinary paragraph text.", "Related work": "More prose", "Method": "Not a table"},
            {"Introduction": "Another paragraph", "Related work": "More prose", "Method": "Still prose"},
        ],
        table_id="false_positive",
        page=1,
        caption=None,
        bbox=[20, 40, 590, 740],
        extraction_method="pdfplumber_text",
        raw={"page_width": 612, "page_height": 792},
    )

    assert _table_quality_acceptable(block, method="pdfplumber_text") is False


def test_numeric_first_row_is_preserved_as_data_not_header() -> None:
    extracted = [
        ["UAE", "0.8501", "0.017", "21.2"],
        ["Baseline", "0.8120", "0.024", "30.1"],
    ]

    assert _detect_header_end(extracted) == 0
