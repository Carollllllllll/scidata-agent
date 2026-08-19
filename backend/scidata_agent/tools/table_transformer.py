from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from scidata_agent.agent.schemas import SourceType, TableBlock, UploadedFile

# Table Transformer is an optional heavy dependency; lazy-load to keep import fast.
try:
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
except Exception:  # pragma: no cover - handled at runtime
    torch = None
    AutoImageProcessor = None
    TableTransformerForObjectDetection = None


_DETECTION_MODEL = "microsoft/table-transformer-detection"
_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"


class TableTransformerExtractor:
    """Extract tables from PDF pages using Microsoft Table Transformer (TATR).

    TATR is a DETR-based model trained on PubTables-1M. It is especially good at
    detecting tables in scientific/financial documents and at recognizing table
    structure (rows/columns) even when grid lines are absent.

    Text is read from the original PDF vectors rather than OCR: TATR gives cell
    bounding boxes in image coordinates, which are mapped back to PDF points and
    cropped with pdfplumber for accurate text extraction.
    """

    def __init__(self, device: str | None = None) -> None:
        if torch is None:
            raise ImportError(
                "Table Transformer requires 'torch', 'torchvision' and 'transformers'. "
                "Install them with: pip install torch torchvision transformers"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_processor: Any = None
        self.det_model: Any = None
        self.str_processor: Any = None
        self.str_model: Any = None

    def _load_models(self) -> None:
        if self.det_model is not None:
            return
        # Use cache_dir env if set, otherwise let transformers use its default.
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        self.det_processor = AutoImageProcessor.from_pretrained(
            _DETECTION_MODEL, cache_dir=cache_dir
        )
        self.det_model = TableTransformerForObjectDetection.from_pretrained(
            _DETECTION_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.det_model.eval()

        self.str_processor = AutoImageProcessor.from_pretrained(
            _STRUCTURE_MODEL, cache_dir=cache_dir
        )
        self.str_model = TableTransformerForObjectDetection.from_pretrained(
            _STRUCTURE_MODEL, cache_dir=cache_dir
        ).to(self.device)
        self.str_model.eval()

    def extract_tables(
        self,
        uploaded: UploadedFile,
        page_numbers: list[int] | None = None,
        dpi: int = 200,
        detection_threshold: float = 0.9,
        structure_threshold: float = 0.7,
    ) -> list[TableBlock]:
        """Extract TableBlocks from selected pages of a PDF."""
        self._load_models()
        pdf_path = str(uploaded.path)
        # Total page count is obtained cheaply via pdfplumber (no PyMuPDF import
        # in this process, avoiding a Windows segfault with torch).
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        pages = page_numbers or list(range(1, total_pages + 1))
        page_images = _render_pages_in_subprocess(pdf_path, pages, dpi)

        all_tables: list[TableBlock] = []
        for page_num in pages:
            image_path = page_images.get(page_num)
            if not image_path or not Path(image_path).exists():
                continue
            image = Image.open(image_path).convert("RGB")
            table_bboxes = self._detect_tables(image, threshold=detection_threshold)
            for bbox in table_bboxes:
                structure = self._recognize_structure(
                    image, bbox, threshold=structure_threshold
                )
                caption = _find_caption_pdfplumber(pdf_path, page_num, bbox, dpi)
                table = self._build_tableblock(
                    uploaded=uploaded,
                    page_number=page_num,
                    image=image,
                    table_bbox=bbox,
                    structure=structure,
                    caption=caption,
                    dpi=dpi,
                )
                if table:
                    all_tables.append(table)
        return all_tables

    def _detect_tables(self, image: Image.Image, threshold: float = 0.9) -> list[list[int]]:
        inputs = self.det_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.det_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.det_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        boxes = []
        for box in results["boxes"]:
            boxes.append([int(round(b.item())) for b in box])
        return boxes

    def _recognize_structure(
        self,
        page_image: Image.Image,
        table_bbox: list[int],
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        margin = 5
        x1 = max(0, table_bbox[0] - margin)
        y1 = max(0, table_bbox[1] - margin)
        x2 = min(page_image.width, table_bbox[2] + margin)
        y2 = min(page_image.height, table_bbox[3] + margin)
        cropped = page_image.crop((x1, y1, x2, y2))

        inputs = self.str_processor(images=cropped, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.str_model(**inputs)
        target_sizes = torch.tensor([cropped.size[::-1]])
        results = self.str_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        rows: list[list[float]] = []
        cols: list[list[float]] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label_name = self.str_model.config.id2label[label.item()].lower()
            box = [b.item() for b in box]
            # Map coordinates back to the original page image.
            box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
            if "spanning" in label_name:
                continue
            if "column" in label_name:
                cols.append(box)
            elif "row" in label_name or "header" in label_name:
                rows.append(box)

        rows.sort(key=lambda b: (b[1] + b[3]) / 2)
        cols.sort(key=lambda b: (b[0] + b[2]) / 2)

        cells: list[dict[str, Any]] = []
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(cols):
                cell_box = _intersect_boxes(row, col)
                if cell_box:
                    cells.append({"row": r_idx, "col": c_idx, "bbox": cell_box})

        return {"rows": rows, "cols": cols, "cells": cells}

    def _build_tableblock(
        self,
        uploaded: UploadedFile,
        page_number: int,
        image: Image.Image,
        table_bbox: list[int],
        structure: dict[str, Any],
        caption: str | None,
        dpi: int,
    ) -> TableBlock | None:
        cells = structure.get("cells", [])
        if not cells:
            return None

        rows = structure.get("rows", [])
        cols = structure.get("cols", [])
        if len(rows) < 2 or len(cols) < 2:
            return None

        # Read text from the PDF for each cell.
        cell_texts: dict[tuple[int, int], str] = {}
        for cell in cells:
            text = self._extract_cell_text_from_pdf(
                uploaded.path, page_number, cell["bbox"], image.size, dpi
            )
            cell_texts[(cell["row"], cell["col"])] = text

        # Detect which top rows are header rows and merge multi-row headers.
        header_row_count = _find_header_boundary(cell_texts, rows, cols)
        header_row_count = max(1, min(header_row_count, len(rows) - 1))
        raw_header = _merge_header_rows(cell_texts, header_row_count, cols)

        # For any still-empty header column, try to read the text from the PDF
        # across the whole header area of that column.
        header = []
        for c_idx, col in enumerate(cols):
            h = raw_header[c_idx].strip()
            if not h:
                header_top = min(rows[r][1] for r in range(header_row_count))
                header_bottom = max(rows[r][3] for r in range(header_row_count))
                h = self._extract_header_text_from_pdf(
                    uploaded.path,
                    page_number,
                    col,
                    header_top,
                    header_bottom,
                    image.size,
                    dpi,
                )
            header.append(h)

        # Ensure unique non-empty headers.
        seen: set[str] = set()
        final_header: list[str] = []
        for i, h in enumerate(header):
            h = str(h).strip()
            if not h or h in seen:
                h = f"column_{i + 1}"
            seen.add(h)
            final_header.append(h)
        header = final_header

        data_rows: list[dict[str, Any]] = []
        for r_idx in range(header_row_count, len(rows)):
            row_dict: dict[str, Any] = {}
            for c_idx, col_name in enumerate(header):
                text = cell_texts.get((r_idx, c_idx), "")
                row_dict[col_name] = _clean_cell_value(text)
            if any(v is not None and str(v).strip() for v in row_dict.values()):
                data_rows.append(row_dict)

        if not data_rows:
            return None

        return TableBlock(
            source_file=uploaded.filename,
            source_path=str(uploaded.path),
            source_type=SourceType.PDF_TABLE,
            columns=header,
            rows=data_rows,
            table_id=f"pdf_table_tatr_{uuid4().hex[:8]}",
            page=page_number,
            caption=caption,
            bbox=_image_bbox_to_pdf_points(table_bbox, dpi),
            extraction_method="table_transformer",
            raw={
                "model": _STRUCTURE_MODEL,
                "dpi": dpi,
                "cell_count": len(cells),
                "row_count": len(rows),
                "column_count": len(cols),
                "header_row_count": header_row_count,
                "header_before_dedup": raw_header,
            },
        )

    def _extract_cell_text_from_pdf(
        self,
        pdf_path: Path,
        page_number: int,
        image_bbox: list[float],
        image_size: tuple[int, int],
        dpi: int,
    ) -> str:
        """Map image bbox to PDF points and extract text with pdfplumber."""
        import pdfplumber

        pdf_bbox = _image_bbox_to_pdf_points(image_bbox, dpi)
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_number - 1]
            page_width = float(page.width)
            page_height = float(page.height)
            # Clamp to page bounds.
            x0 = max(0, pdf_bbox[0])
            y0 = max(0, pdf_bbox[1])
            x1 = min(page_width, pdf_bbox[2])
            y1 = min(page_height, pdf_bbox[3])
            if x1 <= x0 or y1 <= y0:
                return ""
            cropped = page.crop((x0, y0, x1, y1))
            text = cropped.extract_text() or ""
        return text.strip()

    def _extract_header_text_from_pdf(
        self,
        pdf_path: Path,
        page_number: int,
        col_bbox: list[float],
        header_top: float,
        header_bottom: float,
        image_size: tuple[int, int],
        dpi: int,
    ) -> str:
        """Extract text for a header column from the full header vertical span.

        TATR's per-cell bbox may be too tight if the header text is slightly
        outside the detected cell; reading the whole column header area gives
        a second chance to recover the label.
        """
        import pdfplumber

        # Slightly widen the column to catch text near the cell edges.
        x_pad = 4
        pdf_col = _image_bbox_to_pdf_points(col_bbox, dpi)
        pdf_top = header_top * (72.0 / dpi)
        pdf_bottom = header_bottom * (72.0 / dpi)
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_number - 1]
            page_width = float(page.width)
            page_height = float(page.height)
            x0 = max(0, pdf_col[0] - x_pad)
            y0 = max(0, pdf_top)
            x1 = min(page_width, pdf_col[2] + x_pad)
            y1 = min(page_height, pdf_bottom)
            if x1 <= x0 or y1 <= y0:
                return ""
            cropped = page.crop((x0, y0, x1, y1))
            text = cropped.extract_text() or ""
        return text.strip()


def _find_header_boundary(
    cell_texts: dict[tuple[int, int], str],
    rows: list[list[float]],
    cols: list[list[float]],
) -> int:
    """Return the number of top rows that constitute the table header.

    Heuristic: header rows contain mostly text, data rows contain mostly
    numbers. A cell is treated as numeric if every non-empty token inside it is
    a number, which handles stacked numbers rendered in a single cell. Scan from
    the top and stop at the first row where numeric cells dominate. At least one
    row is always treated as header.
    """
    if len(rows) <= 2:
        return 1

    numeric_ratios: list[float] = []
    for r_idx in range(len(rows)):
        text_count = 0
        numeric_count = 0
        empty_count = 0
        for c_idx in range(len(cols)):
            text = cell_texts.get((r_idx, c_idx), "").strip()
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
        ratio = numeric_count / non_empty if non_empty > 0 else 0.0
        numeric_ratios.append(ratio)

    # Find first row where numeric ratio exceeds text ratio; cap at 5 header rows.
    for r_idx in range(1, min(len(rows), 6)):
        if numeric_ratios[r_idx] >= 0.5:
            return r_idx
    return 1


def _merge_header_rows(
    cell_texts: dict[tuple[int, int], str],
    header_row_count: int,
    cols: list[list[float]],
) -> list[str]:
    """Merge the text of consecutive header rows vertically per column."""
    headers: list[str] = []
    for c_idx in range(len(cols)):
        parts: list[str] = []
        for r_idx in range(header_row_count):
            text = cell_texts.get((r_idx, c_idx), "").strip()
            if text and text not in parts:
                parts.append(text)
        headers.append(" ".join(parts))
    return headers


def _classify_cell_text(text: str) -> str:
    """Classify cell text as 'text', 'numeric', or 'empty'.

    Handles stacked numbers and cells that only contain metric arrows/symbols.
    """
    text = text.strip()
    if not text:
        return "empty"
    import re

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


def _intersect_boxes(box_a: list[float], box_b: list[float]) -> list[float] | None:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return None
    return [x_left, y_top, x_right, y_bottom]


def _image_bbox_to_pdf_points(bbox: list[float], dpi: int) -> list[float]:
    scale = 72.0 / dpi
    return [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]


def _render_pages_in_subprocess(pdf_path: str, page_numbers: list[int], dpi: int) -> dict[int, str]:
    """Render selected PDF pages to PNG using a subprocess with only PyMuPDF.

    Loading PyMuPDF and torch in the same Windows process causes a segfault in
    some wheel combinations, so we isolate the rendering step.
    """
    if not page_numbers:
        return {}
    renderer = Path(__file__).with_name("_pdf_renderer.py")
    if not renderer.exists():
        raise FileNotFoundError(f"PDF renderer helper not found: {renderer}")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                str(Path(sys.executable)),
                str(renderer),
                "--pdf",
                pdf_path,
                "--output-dir",
                tmpdir,
                "--pages",
                ",".join(str(p) for p in page_numbers),
                "--dpi",
                str(dpi),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PDF rendering subprocess failed: {result.stderr or result.stdout}"
            )
        mapping: dict[int, str] = {}
        for line in result.stdout.strip().splitlines():
            if "\t" not in line:
                continue
            page_str, path = line.split("\t", 1)
            mapping[int(page_str)] = path
        # The temp dir will be deleted when the context exits, so copy images to
        # a stable location. Using a sibling folder to the PDF is simple and
        # avoids passing large byte strings between processes.
        stable_dir = Path(pdf_path).parent / ".scidata_render"
        stable_dir.mkdir(parents=True, exist_ok=True)
        final: dict[int, str] = {}
        for page_num, src in mapping.items():
            dest = stable_dir / f"{Path(pdf_path).stem}_p{page_num}.png"
            import shutil

            shutil.copy2(src, dest)
            final[page_num] = str(dest)
    return final


def _find_caption_pdfplumber(
    pdf_path: str,
    page_number: int,
    table_image_bbox: list[int],
    dpi: int,
) -> str | None:
    """Find a caption line (Table N / Tab. N) near the table bbox using pdfplumber."""
    import pdfplumber
    import re

    pdf_bbox = _image_bbox_to_pdf_points(table_image_bbox, dpi)
    pattern = re.compile(r"^(?:Table|Tab\.)\s*\d+[:.)\s]", re.IGNORECASE)
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        table_top = pdf_bbox[1]
        table_bottom = pdf_bbox[3]
        words = page.extract_words() or []
        # Group words by line.
        lines: dict[int, list[dict]] = {}
        for word in words:
            top = int(round(float(word.get("top", 0)) / 2.0))
            lines.setdefault(top, []).append(word)

        candidates: list[tuple[float, str]] = []
        for line_words in lines.values():
            line_words = sorted(line_words, key=lambda w: float(w.get("x0", 0)))
            text = " ".join(str(w.get("text", "")).strip() for w in line_words)
            if not pattern.match(text):
                continue
            y = min(float(w.get("top", 0)) for w in line_words)
            candidates.append((y, text))

        above = [(y, t) for y, t in candidates if y < table_top]
        below = [(y, t) for y, t in candidates if y > table_bottom]
        if above:
            return max(above, key=lambda item: item[0])[1]
        if below:
            return min(below, key=lambda item: item[0])[1]
    return None


def _clean_cell_value(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "—", "--", "N/A", "n/a", "NA", "null", "None"}:
        return None
    try:
        if __import__("re").fullmatch(r"[+-]?\d+\.?\d*", text.replace(",", "")):
            num = float(text.replace(",", ""))
            return int(num) if num.is_integer() else num
    except ValueError:
        pass
    return text
