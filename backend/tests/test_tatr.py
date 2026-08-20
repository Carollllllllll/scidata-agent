from __future__ import annotations

import os
from pathlib import Path

import pytest

from scidata_agent.agent.schemas import UploadedFile
from tests.test_pdf_table_extraction import _make_table_pdf


@pytest.mark.tatr
def test_tatr_loads_and_extracts_without_parser_fallback(tmp_path: Path) -> None:
    """Prove the real TATR adapter loads and returns TATR-labelled tables.

    This intentionally bypasses parse_pdf_tables(), whose production behavior
    permits a pdfplumber fallback when TATR is unavailable.
    """
    if os.getenv("RUN_TATR_TESTS", "false").lower() not in {"1", "true", "yes"}:
        pytest.skip("Set RUN_TATR_TESTS=true to load the optional local TATR models.")

    # Import only after the explicit opt-in. Some mismatched native torch builds
    # abort the interpreter during import and therefore cannot be caught by
    # pytest or a normal try/except block.
    from scidata_agent.tools.table_transformer import TableTransformerExtractor

    pdf_path = _make_table_pdf(tmp_path / "tatr_fixture.pdf")
    uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
    extractor = TableTransformerExtractor(device="cpu")

    try:
        extractor._load_models()
        tables = extractor.extract_tables(uploaded, page_numbers=[1], dpi=150)
    except Exception as exc:  # keep the failure actionable in normal pytest output
        pytest.fail(f"TATR environment/model startup failed: {type(exc).__name__}: {exc}")

    assert tables, "TATR loaded but did not detect the known table fixture"
    assert {table.extraction_method for table in tables} == {"table_transformer"}
    assert all(
        table.raw.get("model") == "microsoft/table-transformer-structure-recognition"
        for table in tables
    )
