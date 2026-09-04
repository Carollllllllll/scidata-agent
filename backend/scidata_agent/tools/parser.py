from __future__ import annotations

import html
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

try:
    import pdfplumber
except ImportError:  # Optional for CSV/TSV/Excel-only parsing paths.
    pdfplumber = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except ImportError:  # Optional for CSV/TSV/Excel-only parsing paths.
    PdfReader = None  # type: ignore[assignment,misc]

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:  # Optional for CSV/TSV/Excel-only parsing paths.
    fitz = None  # type: ignore[assignment]

from scidata_agent.agent.schemas import (
    HeadingCandidate,
    ParsedSources,
    SectionBlock,
    SectionPlan,
    SourceType,
    TableBlock,
    TextBlock,
    UploadedFile,
)


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".tsv", ".dat", ".xlsx", ".xls", ".html", ".htm"}
_UNSET = object()
DEFAULT_PDF_PARSE_MAX_WORKERS = 2


def _use_table_transformer() -> bool:
    # TATR is a large optional model.  Keep the default parser deterministic and
    # offline-safe; deployments that have pinned/cached the model can opt in.
    return os.getenv("USE_TABLE_TRANSFORMER", "false").lower() in {"1", "true", "yes"}


def _parse_worker_count(max_workers: int | None) -> int:
    configured = max_workers
    if configured is None:
        try:
            configured = int(os.getenv("SCIDATA_PDF_PARSE_MAX_WORKERS", str(DEFAULT_PDF_PARSE_MAX_WORKERS)))
        except (TypeError, ValueError):
            configured = DEFAULT_PDF_PARSE_MAX_WORKERS
    return max(1, int(configured))


def _parse_one_file(
    uploaded: UploadedFile,
    max_pdf_pages: int | None,
) -> tuple[
    list[TextBlock],
    list[HeadingCandidate],
    list[TableBlock],
    str | None,
    dict[str, Any] | None,
    list[str],
]:
    suffix = uploaded.path.suffix.lower()
    if suffix == ".pdf":
        warnings: list[str] = []
        reader = None
        pdf_document = None
        try:
            if PdfReader is None:
                raise ImportError("PDF text parsing requires the 'pypdf' package.")
            reader = PdfReader(str(uploaded.path))
        except Exception as exc:
            warnings.append(f"PDF reader failed for {uploaded.filename}: {type(exc).__name__}: {exc}")
        try:
            if pdfplumber is not None:
                pdf_document = pdfplumber.open(str(uploaded.path))
        except Exception as exc:
            warnings.append(f"PDF layout reader failed for {uploaded.filename}: {type(exc).__name__}: {exc}")

        pdf_blocks: list[TextBlock] = []
        heading_candidates: list[HeadingCandidate] = []
        tables: list[TableBlock] = []
        title: str | None = None
        table_status: dict[str, Any] | None = None
        try:
            if reader is not None:
                try:
                    pdf_blocks = parse_pdf(uploaded, max_pages=max_pdf_pages, reader=reader)
                except Exception as exc:
                    warnings.append(
                        f"PDF text parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
                    )
            try:
                heading_candidates = extract_heading_candidates(
                    uploaded,
                    pdf_blocks,
                    max_pages=max_pdf_pages,
                    pdf_document=pdf_document,
                )
            except Exception as exc:
                warnings.append(
                    f"PDF heading parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
                )
            try:
                tables, table_status = _parse_pdf_tables_with_status(
                    uploaded,
                    max_pages=max_pdf_pages,
                    pdf_document=pdf_document,
                )
                fallback_reason = table_status.get("fallback_reason")
                if fallback_reason:
                    warnings.append(
                        f"Table Transformer unavailable for {uploaded.filename}; "
                        f"used pdfplumber fallback: {fallback_reason}"
                    )
            except Exception as exc:
                warnings.append(
                    f"PDF table parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
                )
            try:
                title = extract_pdf_title(uploaded.path, reader=reader)
            except Exception as exc:
                warnings.append(
                    f"PDF title parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
                )
        finally:
            if pdf_document is not None:
                try:
                    pdf_document.close()
                except OSError as exc:
                    warnings.append(
                        f"PDF layout close failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
                    )
        return pdf_blocks, heading_candidates, tables, title, table_status, warnings
    if suffix in {".csv", ".tsv", ".dat"}:
        try:
            return [], [], [parse_csv(uploaded)], None, None, []
        except Exception as exc:
            return [], [], [], None, None, [
                f"Table parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
            ]
    if suffix in {".xlsx", ".xls"}:
        try:
            return [], [], parse_excel(uploaded), None, None, []
        except Exception as exc:
            return [], [], [], None, None, [
                f"Workbook parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
            ]
    if suffix in {".html", ".htm"}:
        try:
            text = _html_to_text(uploaded.path.read_text(encoding="utf-8", errors="ignore"))
            return [
                TextBlock(
                    source_file=uploaded.filename,
                    source_path=str(uploaded.path),
                    source_type=SourceType.UNKNOWN,
                    page=None,
                    text=text[:200000],
                    chunk_id=f"{uploaded.path.stem}_html",
                )
            ], [], [], None, None, []
        except Exception as exc:
            return [], [], [], None, None, [
                f"HTML parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
            ]
    return [], [], [], None, None, []


def _html_to_text(value: str) -> str:
    """Conservatively turn a materialized landing page into text evidence."""

    value = re.sub(r"<(script|style|noscript)\b[^>]*>.*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _parse_failure(
    uploaded: UploadedFile,
    exc: Exception,
) -> tuple[
    list[TextBlock],
    list[HeadingCandidate],
    list[TableBlock],
    str | None,
    dict[str, Any] | None,
    list[str],
]:
    return [], [], [], None, None, [
        f"Source parsing failed for {uploaded.filename}: {type(exc).__name__}: {exc}"
    ]


def parse_sources(
    files: list[UploadedFile],
    max_pdf_pages: int | None = None,
    max_workers: int | None = None,
) -> ParsedSources:
    parsed = ParsedSources()
    if not files:
        return parsed

    workers = min(_parse_worker_count(max_workers), len(files))
    if workers == 1:
        outcomes = []
        for uploaded in files:
            try:
                outcomes.append(_parse_one_file(uploaded, max_pdf_pages))
            except Exception as exc:
                outcomes.append(_parse_failure(uploaded, exc))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pdf-parse") as executor:
            futures = [
                executor.submit(_parse_one_file, uploaded, max_pdf_pages)
                for uploaded in files
            ]
            # Consume in file order to keep evidence and exports deterministic.
            outcomes = []
            for uploaded, future in zip(files, futures, strict=True):
                try:
                    outcomes.append(future.result())
                except Exception as exc:
                    outcomes.append(_parse_failure(uploaded, exc))

    for uploaded, outcome in zip(files, outcomes, strict=True):
        text_blocks, heading_candidates, tables, title, table_status, warnings = outcome
        parsed.text_blocks.extend(text_blocks)
        parsed.heading_candidates.extend(heading_candidates)
        parsed.tables.extend(tables)
        if table_status is not None:
            parsed.table_extraction_status.append(table_status)
        parsed.parser_warnings.extend(warnings)
        if title:
            parsed.file_titles[uploaded.filename] = title
    return parsed


def extract_pdf_title(path: str | Path, max_chars: int = 220, *, reader: Any = _UNSET) -> str | None:
    """Extract the full paper title from a PDF.

    First try the PDF metadata title field. If it is missing, fall back to
    PyMuPDF layout-based detection on the first page: merge nearby lines with
    the same large font into title candidates and pick the topmost one that
    looks like a title.
    """
    path = Path(path)
    if reader is _UNSET:
        if PdfReader is None:
            reader = None
        else:
            try:
                reader = PdfReader(str(path))
            except Exception:
                reader = None
    if reader is not None:
        try:
            meta = reader.metadata or {}
            meta_title = meta.get("title") or meta.get("Title")
            if meta_title and isinstance(meta_title, str):
                cleaned = _compact_text(meta_title)
                if _looks_like_title(cleaned):
                    return cleaned[:max_chars]
        except Exception:
            pass

    if fitz is None:
        return None
    doc = None
    try:
        doc = fitz.open(str(path))
        if len(doc) == 0:
            return None
        page = doc[0]
        page_height = page.rect.height
        blocks = page.get_text("dict").get("blocks", [])

        # Group text spans into line records with font size.
        line_records: list[dict] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                chars = [span.get("text", "") for span in line.get("spans", [])]
                text = "".join(chars).strip()
                if not text:
                    continue
                sizes = [
                    float(span.get("size", 0))
                    for span in line.get("spans", [])
                    if isinstance(span.get("size"), int | float)
                ]
                max_size = max(sizes) if sizes else 0.0
                bbox = line.get("bbox", [0, 0, 0, 0])
                line_records.append({
                    "text": text,
                    "top": float(bbox[1]),
                    "bottom": float(bbox[3]),
                    "left": float(bbox[0]),
                    "right": float(bbox[2]),
                    "size": max_size,
                })

        if not line_records:
            return None

        # Merge lines that are vertically close and horizontally overlapping;
        # this handles stylised titles rendered with overlapping large/small fonts.
        merged = _merge_title_lines(line_records)

        sizes = [line["size"] for line in merged if line["size"] > 0]
        median_size = _median(sizes) if sizes else 0.0
        large_size_threshold = median_size + 1.5 if median_size > 0 else 12.0

        candidates: list[tuple[float, str]] = []
        for idx, line in enumerate(merged):
            text = _compact_text(line["text"])
            if not _looks_like_title(text):
                continue
            # Prefer lines in the upper half of the first page.
            if line["top"] > page_height * 0.65:
                continue
            # Collect immediately following large-font title-like lines as a
            # multi-line title (common for long LaTeX titles).
            combined_parts = [text]
            for next_line in merged[idx + 1 :]:
                if next_line["top"] - line["bottom"] > 45.0:
                    break
                if next_line["size"] < large_size_threshold:
                    break
                next_text = _compact_text(next_line["text"])
                if not _looks_like_title(next_text):
                    break
                combined_parts.append(next_text)
                line["bottom"] = next_line["bottom"]
            combined = " ".join(combined_parts)
            score = line["size"]
            if line["size"] >= large_size_threshold:
                score += 3.0
            # Topmost large title gets a bonus.
            score -= line["top"] * 0.005
            # Slight penalty for very long lines to avoid abstract text.
            score -= max(0, len(combined) - 160) * 0.02
            candidates.append((score, combined))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1][:max_chars]
    except Exception:
        pass
    finally:
        if doc is not None:
            doc.close()
    return None


def _merge_title_lines(lines: list[dict]) -> list[dict]:
    """Merge nearby horizontally-overlapping lines into single title lines.

    Some LaTeX papers render titles with alternating large capitals and smaller
    letters on slightly different baselines. This merges those fragments so the
    title can be read as one continuous line.
    """
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda item: (item["top"], item["left"]))
    merged: list[dict] = []
    for line in sorted_lines:
        if not merged:
            merged.append(line)
            continue
        last = merged[-1]
        vertical_gap = abs(line["top"] - last["top"])
        # Horizontal overlap or close alignment.
        h_overlap = min(line["right"], last["right"]) - max(line["left"], last["left"])
        h_span = max(line["right"], last["right"]) - min(line["left"], last["left"])
        if vertical_gap <= 5.0 and (h_overlap > 0 or h_span < max(line["right"] - line["left"], last["right"] - last["left"]) * 2.5):
            # Merge: keep the larger size, span both bboxes, concatenate text.
            text_parts = [last["text"], line["text"]]
            # Reorder by x coordinate if one clearly starts before the other.
            if line["left"] < last["left"] - 10:
                text_parts = [line["text"], last["text"]]
            last["text"] = " ".join(text_parts)
            last["top"] = min(last["top"], line["top"])
            last["bottom"] = max(last["bottom"], line["bottom"])
            last["left"] = min(last["left"], line["left"])
            last["right"] = max(last["right"], line["right"])
            last["size"] = max(last["size"], line["size"])
        else:
            merged.append(line)
    return merged


def _looks_like_title(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if len(text) < 8 or len(text) > 220:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    lowered = text.lower()
    # Exclude obvious non-titles.
    if re.match(r"^(figure|fig\.?|table|tab\.?)\s*\d", lowered):
        return False
    if re.match(r"^\d+(\.\d+)*\s+(introduction|related work|method|experiments|results|discussion|references|appendix)", lowered):
        return False
    if re.match(r"^\[\d+\]", lowered):
        return False
    # Reject arXiv headers and date lines.
    if "arxiv:" in lowered:
        return False
    if re.search(r"\b\d{4}\b.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b.*\b\d{4}\b", lowered):
        return False
    # Titles rarely end with a period and are usually title case or mixed case.
    if text.endswith(".") and len(words) > 6:
        return False
    return True


def parse_pdf(uploaded: UploadedFile, max_pages: int | None = 8, *, reader: Any = _UNSET) -> list[TextBlock]:
    if reader is _UNSET:
        if PdfReader is None:
            raise ImportError("PDF text parsing requires the 'pypdf' package.")
        reader = PdfReader(str(uploaded.path))
    if reader is None:
        raise ValueError("PDF reader is unavailable")
    blocks: list[TextBlock] = []
    total_pages = len(reader.pages)
    page_limit = total_pages if max_pages is None else min(total_pages, max_pages)
    for page_index in range(page_limit):
        page = reader.pages[page_index]
        text = page.extract_text() or ""
        text = _compact_text(text)
        if not text:
            continue
        blocks.extend(
            TextBlock(
                source_file=uploaded.filename,
                source_path=str(uploaded.path),
                source_type=SourceType.PDF_TEXT,
                page=page_index + 1,
                text=chunk,
                chunk_id=f"{uploaded.file_id}_p{page_index + 1}_{uuid4().hex[:6]}",
            )
            for chunk in _chunk_text(text, max_chars=4500)
        )
    return blocks


def parse_pdf_tables(uploaded: UploadedFile, max_pages: int | None = 8) -> list[TableBlock]:
    """Extract structured tables from a PDF.

    Uses Microsoft Table Transformer (TATR) by default for robust detection of
    both ruled and wireless tables. Falls back to pdfplumber-based extraction
    when TATR is unavailable or disabled.
    """
    tables, _ = _parse_pdf_tables_with_status(uploaded, max_pages=max_pages)
    return tables


def _parse_pdf_tables_with_status(
    uploaded: UploadedFile,
    max_pages: int | None = 8,
    *,
    pdf_document: Any | None = None,
) -> tuple[list[TableBlock], dict[str, Any]]:
    status: dict[str, Any] = {
        "source_file": uploaded.filename,
        "tatr_enabled": _use_table_transformer(),
        "method": "pdfplumber",
        "fallback_reason": None,
        "table_count": 0,
    }
    if _use_table_transformer():
        try:
            from scidata_agent.tools.table_transformer import TableTransformerExtractor

            extractor = TableTransformerExtractor()
            if pdfplumber is None:
                raise ImportError("PDF table parsing requires the 'pdfplumber' package.")
            if pdf_document is not None:
                total_pages = len(pdf_document.pages)
            else:
                with pdfplumber.open(str(uploaded.path)) as pdf:
                    total_pages = len(pdf.pages)
            page_limit = total_pages if max_pages is None else min(total_pages, max_pages)
            page_numbers = list(range(1, page_limit + 1))
            tables = extractor.extract_tables(
                uploaded,
                page_numbers=page_numbers,
                pdf_document=pdf_document,
            )
            if tables:
                status["method"] = "table_transformer"
                status["table_count"] = len(tables)
                return tables, status
            status["fallback_reason"] = "Table Transformer detected no usable tables."
        except Exception as exc:
            # Do not hide an optional-model failure: it is important provenance
            # for both operators and downstream quality review.
            status["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    tables = _parse_pdf_tables_pdfplumber(
        uploaded,
        max_pages=max_pages,
        pdf_document=pdf_document,
    )
    status["table_count"] = len(tables)
    return tables, status


def _parse_pdf_tables_pdfplumber(
    uploaded: UploadedFile,
    max_pages: int | None = 8,
    *,
    pdf_document: Any | None = None,
) -> list[TableBlock]:
    """Fallback pdfplumber-based table extraction."""
    if pdfplumber is None:
        return []
    tables: list[TableBlock] = []
    owns_document = pdf_document is None
    pdf = pdf_document
    try:
        if pdf is None:
            pdf = pdfplumber.open(str(uploaded.path))
        total_pages = len(pdf.pages)
        page_limit = total_pages if max_pages is None else min(total_pages, max_pages)
        for page_index in range(page_limit):
            page = pdf.pages[page_index]
            page_number = page_index + 1
            page_tables = _extract_page_tables(page, page_number)
            for table in page_tables:
                table.source_file = uploaded.filename
                table.source_path = str(uploaded.path)
            tables.extend(page_tables)
    except Exception:
        return []
    finally:
        if owns_document and pdf is not None:
            pdf.close()
    return _dedupe_pdf_tables(tables)


def _extract_page_tables(page, page_number: int) -> list[TableBlock]:
    """Extract tables from a single pdfplumber page using multiple strategies."""
    found: list[TableBlock] = []

    strategies = [
        ("pdfplumber_lines", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}),
        ("pdfplumber_lines_strict", {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 2, "join_tolerance": 2}),
        ("pdfplumber_text", {"vertical_strategy": "text", "horizontal_strategy": "text", "min_words_vertical": 3, "min_words_horizontal": 2}),
    ]

    detected_bboxes: list[tuple[float, float, float, float]] = []
    for method, settings in strategies:
        try:
            table_objs = page.find_tables(table_settings=settings) or []
        except Exception:
            continue
        for table_obj in table_objs:
            bbox = list(table_obj.bbox)
            if _bbox_overlaps_existing(bbox, detected_bboxes, iou_threshold=0.7):
                continue
            extracted = table_obj.extract()
            if not extracted or len(extracted) < 2:
                continue
            block = _table_to_tableblock(extracted, page_number, bbox, method)
            if block:
                block.caption = _find_caption_for_table(page, bbox)
                block.raw.update({"page_width": float(page.width), "page_height": float(page.height)})
                if _table_quality_acceptable(block, method=method):
                    detected_bboxes.append(bbox)
                    found.append(block)
    return found


def _find_caption_for_table(page, table_bbox: list[float]) -> str | None:
    """Locate the caption line nearest to the table (above or below)."""
    table_top, table_bottom = table_bbox[1], table_bbox[3]
    page_height = page.height
    words = page.extract_words() or []
    # Group words by line (within 2 points vertically) and concatenate.
    lines: dict[int, list[dict]] = {}
    for word in words:
        top = int(round(float(word.get("top", 0)) / 2.0))
        lines.setdefault(top, []).append(word)

    pattern = re.compile(r"^(?:Table|Tab\.)\s*\d+[:.)\s]", re.IGNORECASE)
    candidates: list[tuple[float, str]] = []
    for line_words in lines.values():
        line_words = sorted(line_words, key=lambda w: float(w.get("x0", 0)))
        text = " ".join(str(w.get("text", "")).strip() for w in line_words)
        if not pattern.match(text):
            continue
        y = min(float(w.get("top", 0)) for w in line_words)
        if y < 0 or y > page_height:
            continue
        candidates.append((y, text))
    if not candidates:
        return None

    # Prefer caption above the table; allow below if none above.
    above = [(y, t) for y, t in candidates if y < table_top]
    below = [(y, t) for y, t in candidates if y > table_bottom]
    if above:
        return max(above, key=lambda item: item[0])[1]
    if below:
        return min(below, key=lambda item: item[0])[1]
    return None


def _table_to_tableblock(
    extracted: list[list[Any]],
    page_number: int,
    bbox: list[float],
    method: str,
) -> TableBlock | None:
    if not extracted or len(extracted) < 2:
        return None

    # Detect header boundary: merge consecutive top rows that are mostly text
    # until a row dominated by numbers is found.
    header_end = _detect_header_end(extracted)
    header_rows = extracted[:header_end]
    data_rows_raw = extracted[header_end:]

    # Build merged header per column.
    max_cols = max(len(row) for row in extracted) if extracted else 0
    header: list[str] = []
    for col_idx in range(max_cols):
        parts: list[str] = []
        for row in header_rows:
            cell = row[col_idx] if col_idx < len(row) else None
            text = str(cell).strip() if cell is not None else ""
            if text and text not in parts:
                parts.append(text)
        header.append(" ".join(parts))

    if not any(header):
        header = [f"column_{index + 1}" for index in range(max_cols)]

    # Ensure unique non-empty headers.
    seen: set[str] = set()
    final_header: list[str] = []
    for i, h in enumerate(header):
        h = h.strip()
        if not h or h in seen:
            h = f"column_{i + 1}"
        seen.add(h)
        final_header.append(h)
    header = final_header

    rows = []
    for row in data_rows_raw:
        row_dict = {}
        for col_index, column in enumerate(header):
            value = row[col_index] if col_index < len(row) else None
            row_dict[column] = _clean_cell_value(value)
        if any(v is not None and str(v).strip() for v in row_dict.values()):
            rows.append(row_dict)
    if not rows:
        return None

    return TableBlock(
        source_file="",  # filled by caller
        source_path="",  # filled by caller
        source_type=SourceType.PDF_TABLE,
        columns=header,
        rows=rows,
        table_id=f"pdf_table_{uuid4().hex[:8]}",
        page=page_number,
        bbox=bbox,
        extraction_method=method,
        raw={
            "bbox": bbox,
            "row_count": len(rows),
            "column_count": len(header),
            "header_row_count": len(header_rows),
            "header_inferred": bool(header_rows),
        },
    )


def _detect_header_end(extracted: list[list[Any]]) -> int:
    """Return the index just past the header rows in a pdfplumber extraction.

    Header rows are expected to contain mostly text; data rows mostly numbers.
    If the first row is already numeric-dominated, return zero so callers keep
    it as data and generate neutral column names instead of inventing headers.
    """
    if len(extracted) < 2:
        return 1

    for r_idx in range(min(6, len(extracted))):
        text_count = 0
        numeric_count = 0
        empty_count = 0
        for cell in extracted[r_idx]:
            text = str(cell).strip() if cell is not None else ""
            if not text:
                empty_count += 1
                continue
            classification = _classify_cell_text(text)
            if classification == "numeric":
                numeric_count += 1
            elif classification == "text":
                text_count += 1
            else:
                empty_count += 1
        non_empty = text_count + numeric_count
        if non_empty > 0 and numeric_count / non_empty >= 0.5:
            return r_idx
    return 1


def _classify_cell_text(text: str) -> Literal["text", "numeric", "empty"]:
    """Classify a cell as text, numeric, or empty.

    A cell is numeric if every non-empty token looks like a number (after
    stripping arrows and common metric suffixes). A cell is text if it contains
    any clear text token. This handles stacked numbers in a single cell.
    """
    text = text.strip()
    if not text:
        return "empty"
    # Strip common table decorations and split into tokens.
    cleaned = re.sub(r"^[↑↓⇅\+\-]?\s*", "", text)
    cleaned = re.sub(r"\s*[↑↓⇅]?\s*$", "", cleaned)
    tokens = [t.strip() for t in re.split(r"[\s\n]+", cleaned) if t.strip()]
    if not tokens:
        return "empty"
    numeric_tokens = 0
    text_tokens = 0
    for token in tokens:
        t = token.replace(",", "")
        if re.fullmatch(r"[+-]?\d+\.?\d*", t):
            numeric_tokens += 1
        elif t in {"-", "—", "--", "/", "N/A", "n/a", "NA"}:
            numeric_tokens += 1
        else:
            text_tokens += 1
    if text_tokens > 0:
        return "text"
    return "numeric"


def _clean_cell_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "—", "--", "N/A", "n/a", "NA", "null", "None"}:
        return None
    # Try to keep numeric values as numbers for downstream extraction.
    try:
        if re.fullmatch(r"[+-]?\d+\.?\d*", text.replace(",", "")):
            num = float(text.replace(",", ""))
            return int(num) if num.is_integer() else num
    except ValueError:
        pass
    return text


def _table_quality_acceptable(block: TableBlock, method: str | None = None) -> bool:
    """Reject accidental table detections (page headers, text columns, tiny fragments)."""
    rows = block.rows
    columns = block.columns
    if len(columns) < 2 or len(rows) < 2:
        return False

    # Text-strategy detections are aggressive on multi-column layouts; be strict.
    non_empty_headers = [str(col).strip() for col in columns if str(col).strip()]
    header_fill_ratio = len(non_empty_headers) / len(columns) if columns else 0.0
    if method == "pdfplumber_text":
        has_caption = bool(block.caption and str(block.caption).strip())
        # Text-based table finding eagerly interprets ordinary multi-column
        # papers as tables.  Scientific tables should have a nearby caption;
        # without one, prefer a false negative to fabricated structured data.
        if not has_caption:
            return False
        min_rows = 2
        min_cols = 2
        if len(columns) < min_cols or len(rows) < min_rows:
            return False
        if header_fill_ratio < 0.8:
            return False
        if any(len(col) > 35 for col in non_empty_headers):
            return False

        page_width = float(block.raw.get("page_width") or 0)
        page_height = float(block.raw.get("page_height") or 0)
        if block.bbox and page_width > 0 and page_height > 0:
            x0, y0, x1, y1 = block.bbox
            area_ratio = max(0.0, (x1 - x0) * (y1 - y0)) / (page_width * page_height)
            block.raw["page_area_ratio"] = round(area_ratio, 4)
            if area_ratio > 0.62:
                return False

    total_cells = len(columns) * len(rows)
    empty_cells = sum(1 for row in rows for value in row.values() if value is None or str(value).strip() == "")
    if total_cells > 0 and empty_cells / total_cells > 0.55:
        return False

    # Header should look like headers, not sentence fragments.
    max_header_len = max(len(str(col).strip()) for col in columns)
    if max_header_len > 80:
        return False
    if not any(str(col).strip() for col in columns):
        return False

    # Require at least two rows with multiple non-empty cells.
    dense_rows = sum(1 for row in rows if sum(1 for v in row.values() if v is not None and str(v).strip()) >= 2)
    if dense_rows < 2:
        return False

    return True


def _bbox_overlaps_existing(bbox: list[float], existing: list[list[float]], iou_threshold: float = 0.7) -> bool:
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    for other in existing:
        ox0, oy0, ox1, oy1 = other
        inter_x0, inter_y0 = max(x0, ox0), max(y0, oy0)
        inter_x1, inter_y1 = min(x1, ox1), min(y1, oy1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            continue
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        union_area = area + (ox1 - ox0) * (oy1 - oy0) - inter_area
        if union_area > 0 and inter_area / union_area >= iou_threshold:
            return True
    return False


def _dedupe_pdf_tables(tables: list[TableBlock]) -> list[TableBlock]:
    """Remove tables that are detected twice with nearly identical content."""
    seen: set[str] = set()
    result: list[TableBlock] = []
    for table in tables:
        fingerprint = _table_fingerprint(table)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(table)
    return result


def _table_fingerprint(table: TableBlock) -> str:
    parts = ["|".join(table.columns[:6])]
    for row in table.rows[:3]:
        parts.append("|".join(str(row.get(col, ""))[:40] for col in table.columns[:6]))
    return "@@".join(parts)


def parse_csv(uploaded: UploadedFile) -> TableBlock:
    suffix = uploaded.path.suffix.lower()
    separator = "\t" if suffix == ".tsv" else r"\s+" if suffix == ".dat" else ","
    kwargs: dict[str, Any] = {"sep": separator}
    if suffix == ".dat":
        # Astronomy survey releases conventionally use whitespace columns and
        # comment-prefixed metadata before the header row.
        kwargs.update({"comment": "#", "engine": "python"})
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            dataframe = pd.read_csv(uploaded.path, encoding=encoding, **kwargs)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    return _dataframe_to_table(uploaded, dataframe, SourceType.CSV, uploaded.path.stem)


def parse_excel(uploaded: UploadedFile) -> list[TableBlock]:
    workbook = pd.read_excel(uploaded.path, sheet_name=None)
    tables: list[TableBlock] = []
    for sheet_name, dataframe in workbook.items():
        tables.append(_dataframe_to_table(uploaded, dataframe, SourceType.EXCEL, sheet_name))
    return tables


def extract_heading_candidates(
    uploaded: UploadedFile,
    text_blocks: list[TextBlock],
    max_pages: int | None = 8,
    max_candidates_per_file: int = 80,
    *,
    pdf_document: Any | None = None,
) -> list[HeadingCandidate]:
    candidates = _extract_layout_heading_candidates(
        uploaded,
        max_pages=max_pages,
        pdf_document=pdf_document,
    )
    if not candidates:
        candidates = _extract_text_heading_candidates(text_blocks)
    return sorted(candidates, key=lambda candidate: (candidate.page, candidate.line_index, -candidate.score))[
        :max_candidates_per_file
    ]


def build_section_blocks_from_plan(
    text_blocks: list[TextBlock],
    section_plan: SectionPlan | None,
    max_chars: int = 5500,
) -> list[SectionBlock]:
    if not text_blocks:
        return []
    blocks_by_file: dict[str, list[TextBlock]] = {}
    for block in text_blocks:
        blocks_by_file.setdefault(block.source_file, []).append(block)

    section_blocks: list[SectionBlock] = []
    for source_file, file_blocks in blocks_by_file.items():
        file_blocks = sorted(file_blocks, key=lambda block: (block.page or 0, block.chunk_id))
        sections = [
            section
            for section in (section_plan.sections if section_plan else [])
            if section.source_file in (None, source_file)
        ]
        sections = _dedupe_sections(sections)
        if not sections:
            section_blocks.extend(_fallback_page_section_blocks(file_blocks, max_chars=max_chars))
            continue

        page_text = _page_texts(file_blocks)
        anchors = [_locate_section_anchor(section, page_text) for section in sections]
        anchors = [(section, loc) for section, loc in zip(sections, anchors) if loc is not None]
        anchors.sort(key=lambda item: (item[1][0], item[1][1]))
        if not anchors:
            section_blocks.extend(_fallback_page_section_blocks(file_blocks, max_chars=max_chars))
            continue

        for index, (section, (start_page, start_offset)) in enumerate(anchors):
            if index + 1 < len(anchors):
                end_page, end_offset = anchors[index + 1][1]
            else:
                end_page = max(page_text)
                end_offset = len(page_text[end_page])
            page_end = end_page if end_offset > 0 else max(start_page, end_page - 1)
            page_parts = _slice_page_parts(page_text, start_page, start_offset, end_page, end_offset)
            chunk_index = 0
            for page_number, page_part in page_parts:
                text = _compact_text(page_part)
                if not text:
                    continue
                for chunk in _chunk_text(text, max_chars=max_chars):
                    chunk_index += 1
                    section_blocks.append(
                        SectionBlock(
                            source_file=source_file,
                            source_path=file_blocks[0].source_path,
                            source_type=file_blocks[0].source_type,
                            section_title=section.section_title,
                            section_type=section.section_type,
                            page_start=page_number,
                            page_end=page_number,
                            page=page_number,
                            text=chunk,
                            chunk_id=f"{file_blocks[0].source_file}_sec{index + 1}_{chunk_index}_{uuid4().hex[:6]}",
                            confidence=section.confidence,
                            raw={
                                "start_anchor": section.start_anchor,
                                "section_reason": section.reason,
                                "section_page_start": start_page,
                                "section_page_end": page_end,
                                "used_llm_section_plan": section_plan.used_llm if section_plan else False,
                            },
                        )
                    )
    return section_blocks


def fallback_section_plan_from_candidates(candidates: list[HeadingCandidate]) -> SectionPlan:
    sections = []
    for candidate in candidates:
        section_type = _fallback_section_type(candidate.text)
        if section_type is None:
            continue
        sections.append(
            {
                "section_title": candidate.text,
                "source_file": candidate.source_file,
                "section_type": section_type,
                "start_page": candidate.page,
                "start_anchor": candidate.text,
                "confidence": min(0.75, max(0.35, candidate.score / 10)),
                "reason": "Deterministic fallback inferred section type from heading-like text.",
            }
        )
    return SectionPlan.model_validate(
        {
            "sections": sections,
            "ignored_candidates": [],
            "warnings": ["LLM section interpretation was unavailable; deterministic section fallback was used."],
            "used_llm": False,
        }
    )


def _extract_layout_heading_candidates(
    uploaded: UploadedFile,
    max_pages: int | None,
    *,
    pdf_document: Any | None = None,
) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    if pdfplumber is None:
        return candidates
    owns_document = pdf_document is None
    pdf = pdf_document
    try:
        if pdf is None:
            pdf = pdfplumber.open(str(uploaded.path))
        page_limit = len(pdf.pages) if max_pages is None else min(len(pdf.pages), max_pages)
        for page_index in range(page_limit):
            page = pdf.pages[page_index]
            words = page.extract_words(extra_attrs=["fontname", "size"], keep_blank_chars=False) or []
            lines = _words_to_lines(words)
            if not lines:
                continue
            sizes = [word.get("size") for word in words if isinstance(word.get("size"), int | float)]
            median_size = _median(sizes) if sizes else 0.0
            for line_index, line in enumerate(lines):
                text = _compact_heading_text(line["text"])
                if not text:
                    continue
                font_size = float(line.get("font_size") or 0.0)
                is_bold = bool(line.get("is_bold"))
                score = _heading_candidate_score(text, line_index, font_size, median_size, is_bold)
                if score < 2.5:
                    continue
                before = " ".join(item["text"] for item in lines[max(0, line_index - 2):line_index])
                after = " ".join(item["text"] for item in lines[line_index + 1:line_index + 4])
                candidates.append(
                    HeadingCandidate(
                        source_file=uploaded.filename,
                        source_path=str(uploaded.path),
                        page=page_index + 1,
                        line_index=line_index,
                        text=text,
                        before_text=before[:800] or None,
                        after_text=after[:1200] or None,
                        font_size=font_size or None,
                        is_bold=is_bold,
                        y_position=line.get("top"),
                        extraction_method="layout",
                        score=score,
                    )
                )
    except Exception:
        return []
    finally:
        if owns_document and pdf is not None:
            pdf.close()
    return _dedupe_candidates(candidates)


def _extract_text_heading_candidates(text_blocks: list[TextBlock]) -> list[HeadingCandidate]:
    candidates: list[HeadingCandidate] = []
    for block in text_blocks:
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        for line_index, line in enumerate(lines[:120]):
            text = _compact_heading_text(line)
            score = _heading_candidate_score(text, line_index, 0.0, 0.0, False)
            if score < 2.5:
                continue
            before = " ".join(lines[max(0, line_index - 2):line_index])
            after = " ".join(lines[line_index + 1:line_index + 4])
            candidates.append(
                HeadingCandidate(
                    source_file=block.source_file,
                    source_path=block.source_path,
                    page=block.page or 1,
                    line_index=line_index,
                    text=text,
                    before_text=before[:800] or None,
                    after_text=after[:1200] or None,
                    extraction_method="text",
                    score=score,
                )
            )
    return _dedupe_candidates(candidates)


def _words_to_lines(words: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for word in words:
        top = int(round(float(word.get("top", 0)) / 3.0))
        grouped.setdefault((int(word.get("doctop", word.get("top", 0))) // 1000, top), []).append(word)
    lines = []
    for _, line_words in sorted(grouped.items(), key=lambda item: min(float(word.get("top", 0)) for word in item[1])):
        ordered = sorted(line_words, key=lambda word: float(word.get("x0", 0)))
        text = " ".join(str(word.get("text", "")) for word in ordered).strip()
        if not text:
            continue
        sizes = [float(word.get("size", 0)) for word in ordered if isinstance(word.get("size"), int | float)]
        fonts = [str(word.get("fontname", "")) for word in ordered]
        lines.append(
            {
                "text": text,
                "font_size": max(sizes) if sizes else 0.0,
                "is_bold": any("bold" in font.lower() for font in fonts),
                "top": min(float(word.get("top", 0)) for word in ordered),
            }
        )
    return lines


def _heading_candidate_score(text: str, line_index: int, font_size: float, median_size: float, is_bold: bool) -> float:
    if not text or len(text) > 160:
        return 0.0
    lowered = text.lower().strip()
    if len(text.split()) > 14 and not re.match(r"^\d+(\.\d+)*\s+", text):
        return 0.0
    if re.match(r"^(figure|fig\.|table)\s+\d+", lowered):
        return 0.0
    if re.match(r"^\[\d+\]", lowered):
        return 0.0

    score = 0.0
    if re.match(r"^(\d+|[ivxlcdm]+)(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,:/()&-]{2,}$", text):
        score += 4.0
    if re.match(r"^[A-Z][A-Za-z0-9 ,:/()&-]{2,}$", text) and len(text.split()) <= 8:
        score += 2.0
    if lowered in {"abstract", "introduction", "references", "acknowledgements", "acknowledgments"}:
        score += 5.0
    if any(token in lowered for token in ["method", "approach", "experiment", "result", "evaluation", "discussion", "conclusion", "appendix"]):
        score += 2.0
    if median_size and font_size >= median_size + 1.0:
        score += 2.0
    if is_bold:
        score += 1.5
    if line_index <= 6:
        score += 0.8
    if text.endswith(".") and len(text.split()) > 4:
        score -= 1.0
    return score


def _fallback_page_section_blocks(file_blocks: list[TextBlock], max_chars: int) -> list[SectionBlock]:
    section_blocks: list[SectionBlock] = []
    for block in file_blocks:
        for index, chunk in enumerate(_chunk_text(block.text, max_chars=max_chars), start=1):
            section_blocks.append(
                SectionBlock(
                    source_file=block.source_file,
                    source_path=block.source_path,
                    source_type=block.source_type,
                    section_title=f"Page {block.page}" if block.page else None,
                    section_type="unknown",
                    page_start=block.page,
                    page_end=block.page,
                    page=block.page,
                    text=chunk,
                    chunk_id=f"{block.chunk_id}_fallback_section_{index}",
                    confidence=0.35,
                    raw={"fallback": "page_as_section"},
                )
            )
    return section_blocks


def _fallback_section_type(title: str) -> str | None:
    text = title.lower()
    if "abstract" == text.strip():
        return "abstract"
    if "intro" in text:
        return "introduction"
    if "related" in text or "background" in text:
        return "related_work"
    if any(token in text for token in ["method", "approach", "framework", "architecture", "model"]):
        return "method"
    if any(token in text for token in ["data", "dataset", "observation", "cohort", "sample"]):
        return "data"
    if any(token in text for token in ["experiment", "implementation", "setup", "training"]):
        return "experiments"
    if any(token in text for token in ["result", "evaluation", "performance", "comparison"]):
        return "results"
    if "ablation" in text:
        return "ablation"
    if "discussion" in text:
        return "discussion"
    if "limitation" in text:
        return "limitations"
    if "conclusion" in text:
        return "conclusion"
    if "reference" in text:
        return "references"
    if "appendix" in text or "supplement" in text:
        return "appendix"
    if re.match(r"^(\d+|[ivxlcdm]+)(\.\d+)*\.?\s+", text):
        return "other"
    return None


def _page_texts(file_blocks: list[TextBlock]) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    for block in file_blocks:
        if block.page is None:
            continue
        pages.setdefault(block.page, []).append(block.text)
    return {page: "\n".join(chunks) for page, chunks in pages.items()}


def _locate_section_anchor(section, page_text: dict[int, str]) -> tuple[int, int] | None:
    pages = sorted(page for page in page_text if page >= section.start_page)
    for page in pages:
        offset = _find_anchor_offset(page_text[page], section.start_anchor)
        if offset is not None:
            return page, offset
    return None


def _slice_pages(page_text: dict[int, str], start_page: int, start_offset: int, end_page: int, end_offset: int) -> str:
    return "\n".join(
        text for _, text in _slice_page_parts(page_text, start_page, start_offset, end_page, end_offset)
    )


def _slice_page_parts(
    page_text: dict[int, str],
    start_page: int,
    start_offset: int,
    end_page: int,
    end_offset: int,
) -> list[tuple[int, str]]:
    parts: list[tuple[int, str]] = []
    for page in range(start_page, end_page + 1):
        text = page_text.get(page, "")
        if not text:
            continue
        begin = start_offset if page == start_page else 0
        end = end_offset if page == end_page else len(text)
        parts.append((page, text[begin:end]))
    return parts


def _find_anchor_offset(text: str, anchor: str) -> int | None:
    """Find an LLM-provided heading despite PDF whitespace/punctuation drift."""
    normalized_text, offsets = _normalise_anchor_with_offsets(text)
    for variant in _anchor_variants(anchor):
        exact = text.casefold().find(variant.casefold())
        if exact >= 0:
            return exact

        normalized_anchor, _ = _normalise_anchor_with_offsets(variant)
        if len(normalized_anchor) < 4:
            continue
        normalized_offset = normalized_text.find(normalized_anchor)
        if normalized_offset >= 0:
            return offsets[normalized_offset]
    return None


def _anchor_variants(anchor: str) -> list[str]:
    """Recover the actual trailing heading from layout-merged LLM anchors."""
    anchor = anchor.strip()
    if not anchor:
        return []
    variants = [anchor]
    numbered_heading = re.search(
        r"(?<!\d)(\d+(?:\.\d+)*\.?\s*[A-Z][A-Za-z0-9][A-Za-z0-9 &:/()_-]{1,80})\s*$",
        anchor,
    )
    if numbered_heading:
        variants.append(numbered_heading.group(1).strip())
    return list(dict.fromkeys(variants))


def _normalise_anchor_with_offsets(value: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(value.casefold()):
        if character.isalnum():
            characters.append(character)
            offsets.append(offset)
    return "".join(characters), offsets


def _anchor_exists_in_blocks(anchor: str, blocks: list[TextBlock], start_page: int) -> bool:
    return any(
        (block.page or 0) >= start_page and _find_anchor_offset(block.text, anchor) is not None
        for block in blocks
    )


def _dedupe_sections(sections) -> list:
    result = []
    seen = set()
    for section in sorted(sections, key=lambda item: (item.start_page, item.start_anchor.casefold())):
        normalized_anchor, _ = _normalise_anchor_with_offsets(section.start_anchor)
        key = (section.start_page, normalized_anchor)
        if key in seen:
            continue
        seen.add(key)
        result.append(section)
    return result


def _dedupe_candidates(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    result = []
    seen = set()
    for candidate in sorted(candidates, key=lambda item: (item.page, item.line_index, -item.score)):
        key = (candidate.source_file, candidate.page, candidate.text.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _compact_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip(" .")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _dataframe_to_table(uploaded: UploadedFile, dataframe: pd.DataFrame, source_type: SourceType, table_name: str) -> TableBlock:
    dataframe = dataframe.where(pd.notnull(dataframe), None)
    rows = []
    for record in dataframe.to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                value = None
            cleaned[str(key)] = value
        rows.append(cleaned)
    return TableBlock(
        source_file=uploaded.filename,
        source_path=str(uploaded.path),
        source_type=source_type,
        columns=[str(column) for column in dataframe.columns],
        rows=rows,
        table_id=f"{uploaded.file_id}_{table_name}_{uuid4().hex[:6]}",
    )


def _compact_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    return "\n".join(line for line in lines if line)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n"):
        para_len = len(paragraph) + 1
        if current and current_len + para_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += para_len
    if current:
        chunks.append("\n".join(current))
    return chunks
