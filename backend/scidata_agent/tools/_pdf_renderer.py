from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf as fitz


def render_pages(pdf_path: str, output_dir: str, page_numbers: list[int], dpi: int) -> dict[int, str]:
    """Render selected PDF pages to PNG images in a clean subprocess.

    Running this in a separate process avoids a Windows segfault that occurs
    when PyMuPDF and the torch CPU wheel are loaded in the same interpreter.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        paths: dict[int, str] = {}
        for page_num in page_numbers:
            if page_num < 1 or page_num > len(doc):
                continue
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=dpi)
            out_path = out_dir / f"page_{page_num}.png"
            pix.save(str(out_path))
            paths[page_num] = str(out_path)
        return paths
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render PDF pages to PNG images")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG files")
    parser.add_argument("--pages", required=True, help="Comma-separated 1-based page numbers")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI")
    args = parser.parse_args()

    page_numbers = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
    paths = render_pages(args.pdf, args.output_dir, page_numbers, args.dpi)
    # Print one mapping per line so the parent can parse it robustly.
    for page_num, path in sorted(paths.items()):
        print(f"{page_num}\t{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
