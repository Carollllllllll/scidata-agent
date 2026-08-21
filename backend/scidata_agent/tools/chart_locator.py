from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from scidata_agent.agent.schemas import FigureAsset, UploadedFile

# Caption starters such as "Figure 3:", "Fig. 2(a)", "FIGURE 1." at a line start.
_CAPTION_RE = re.compile(
    r"(?im)^[ \t]*(?P<label>(?:figure|fig\.?)\s*(?P<num>\d+[a-z]?))[ \t]*[:.\-][ \t]*(?P<body>.+)$"
)

# Minimum plot-region size in PDF points; smaller graphic clusters are treated
# as logos/equations rather than charts.
_MIN_REGION_WIDTH = 120.0
_MIN_REGION_HEIGHT = 90.0
# Padding added around the detected graphics so axis tick labels are included.
_REGION_PAD = 28.0
_DEFAULT_RENDER_DPI = 200


def locate_figures(
    uploaded: UploadedFile,
    figures_dir: str | Path,
    max_pages: int | None = None,
    max_figures: int | None = None,
    render_dpi: int = _DEFAULT_RENDER_DPI,
) -> list[FigureAsset]:
    """Locate figures in a PDF by caption + page graphics, render each to PNG.

    The strategy is deterministic (no LLM calls):

    1. Find "Figure N / Fig. N" caption blocks via text blocks on each page.
    2. Collect embedded-image rects and vector-drawing rects above the caption.
    3. Union the graphics into a region, expand it to swallow axis tick labels,
       and render that clip to a PNG with PyMuPDF.

    If ``max_figures`` is provided, only that many figures are processed.
    """
    if uploaded.path.suffix.lower() != ".pdf":
        return []
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    assets: list[FigureAsset] = []
    try:
        document = fitz.open(str(uploaded.path))
    except Exception:
        return []

    try:
        page_total = len(document)
        page_limit = page_total if max_pages is None else min(page_total, max_pages)
        for page_index in range(page_limit):
            if max_figures is not None and max_figures > 0 and len(assets) >= max_figures:
                break
            page = document[page_index]
            for caption in _find_captions(page):
                if max_figures is not None and max_figures > 0 and len(assets) >= max_figures:
                    break
                region, method = _figure_region(page, caption["top"])
                if region is None or region.height < _MIN_REGION_HEIGHT * 0.5:
                    continue
                image_path = _render_region(
                    page,
                    region,
                    figures_path,
                    f"{uploaded.path.stem}_p{page_index + 1}_{caption['label'].replace(' ', '_').replace('.', '')}",
                    dpi=render_dpi,
                )
                if image_path is None:
                    continue
                assets.append(
                    FigureAsset(
                        source_file=uploaded.filename,
                        source_path=str(uploaded.path),
                        page=page_index + 1,
                        label=caption["label"],
                        caption=caption["text"],
                        bbox=[region.x0, region.y0, region.x1, region.y1],
                        image_path=str(image_path),
                        detection_method=method,
                    )
                )
    finally:
        document.close()
    return assets


def _find_captions(page: "fitz.Page") -> list[dict]:
    """Return caption dicts {label, text, top} found in the page's text blocks."""
    captions: list[dict] = []
    for block in page.get_text("blocks"):
        # block: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        if len(block) < 7 or block[6] != 0:
            continue
        text = str(block[4])
        match = _CAPTION_RE.search(text)
        if not match:
            continue
        body = match.group("body")
        caption_text = f"{match.group('label')}: {body}"
        # Append the remainder of the block after the first caption line.
        tail = text[match.end():].strip()
        if tail and not _CAPTION_RE.search(tail):
            caption_text = f"{caption_text} {_compact(tail)}"
        captions.append(
            {
                "label": _normalize_label(match.group("label")),
                "text": _compact(caption_text)[:1500],
                "top": float(block[1]),
            }
        )
    captions.sort(key=lambda item: item["top"])
    return captions


def _figure_region(page: "fitz.Page", caption_top: float) -> tuple["fitz.Rect | None", str]:
    """Compute the figure region above a caption from images and drawings."""
    page_rect = page.rect
    graphics: list[fitz.Rect] = []
    has_image = False

    seen_images: set[int] = set()
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        if xref in seen_images:
            continue
        seen_images.add(xref)
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if rect.y1 <= caption_top + 6 and rect.width > 40 and rect.height > 40:
                graphics.append(rect)
                has_image = True

    drawing_count = 0
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        if rect.y1 <= caption_top + 6 and rect.width > 20 and rect.height > 20:
            drawing_count += 1
            # Only sizeable drawing clusters count as chart candidates.
            if rect.width > 60 and rect.height > 60:
                graphics.append(rect)

    if not graphics:
        # Fallback heuristic: the vertical band directly above the caption.
        fallback = fitz.Rect(
            page_rect.x0 + 36,
            max(page_rect.y0 + 36, caption_top - 420),
            page_rect.x1 - 36,
            caption_top - 4,
        )
        if fallback.height < _MIN_REGION_HEIGHT:
            return None, "none"
        return fallback, "caption_fallback"

    region = graphics[0]
    for rect in graphics[1:]:
        region = region | rect

    if region.width < _MIN_REGION_WIDTH or region.height < _MIN_REGION_HEIGHT:
        if drawing_count < 3 and not has_image:
            return None, "none"

    # Expand so nearby axis tick labels / axis titles are included. Horizontal
    # growth is clamped to the graphics span plus padding: in two-column
    # papers this prevents body text from the adjacent column being merged
    # into the figure region.
    x_lo = max(page_rect.x0, region.x0 - _REGION_PAD)
    x_hi = min(page_rect.x1, region.x1 + _REGION_PAD)
    padded = fitz.Rect(
        x_lo,
        max(page_rect.y0, region.y0 - _REGION_PAD),
        x_hi,
        min(caption_top - 2, region.y1 + _REGION_PAD),
    )
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        block_rect = fitz.Rect(block[:4])
        if block_rect.y1 > caption_top + 2 or not block_rect.intersects(padded):
            continue
        clamped = fitz.Rect(
            max(x_lo, block_rect.x0),
            block_rect.y0,
            min(x_hi, block_rect.x1),
            min(block_rect.y1, caption_top - 2),
        )
        if clamped.width > 0 and clamped.height > 0:
            padded = padded | clamped

    padded = padded & page_rect
    if padded.height < _MIN_REGION_HEIGHT * 0.5:
        return None, "none"
    method = "caption+graphics" if has_image or drawing_count >= 3 else "caption"
    return padded, method


def _render_region(
    page: "fitz.Page",
    region: "fitz.Rect",
    figures_dir: Path,
    name: str,
    dpi: int,
) -> Path | None:
    try:
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=region, alpha=False)
        if pixmap.width < 80 or pixmap.height < 60:
            return None
        output = figures_dir / f"{name}.png"
        pixmap.save(str(output))
        return output
    except Exception:
        return None


def _normalize_label(label: str) -> str:
    match = re.match(r"(?i)(?:figure|fig\.?)\s*(\d+[a-z]?)", label.strip())
    return f"Figure {match.group(1)}" if match else label.strip()


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
