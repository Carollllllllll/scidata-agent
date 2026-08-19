from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# When this file is launched as ``python scripts/verify_tatr.py``, Python adds
# ``scripts`` to sys.path rather than the backend directory. Add the project
# package root explicitly so the test does not depend on PYTHONPATH or cwd.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _parse_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        page = int(item)
        if page < 1:
            raise ValueError("page numbers must be >= 1")
        pages.append(page)
    return pages or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strict TATR smoke test. This calls TableTransformerExtractor directly "
            "and never uses parser.py's pdfplumber fallback."
        )
    )
    parser.add_argument("--pdf", required=True, help="PDF file to test")
    parser.add_argument("--pages", help="1-based comma-separated pages, e.g. 1,2")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--require-table",
        action="store_true",
        help="Fail unless TATR returns at least one structured table",
    )
    parser.add_argument("--output-json", help="Optional path for a machine-readable report")
    args = parser.parse_args()

    started = time.perf_counter()
    report: dict[str, object] = {
        "status": "failed",
        "pdf": str(Path(args.pdf).expanduser().resolve()),
        "pages": _parse_pages(args.pages),
        "dpi": args.dpi,
        "used_fallback": False,
        "fallback_available": False,
    }

    try:
        pdf_path = Path(args.pdf).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
        if args.dpi <= 0:
            raise ValueError("dpi must be positive")

        # Import the project adapter only after validating the input. The adapter
        # loads TATR lazily, so this test exercises the real model startup path.
        from scidata_agent.agent.schemas import UploadedFile
        from scidata_agent.tools.table_transformer import TableTransformerExtractor

        report["python"] = sys.executable
        report["hf_home"] = os.getenv("HF_HOME")
        report["hf_hub_offline"] = os.getenv("HF_HUB_OFFLINE")

        uploaded = UploadedFile(filename=pdf_path.name, path=pdf_path)
        extractor = TableTransformerExtractor(device="cpu")
        extractor._load_models()
        report["models_loaded"] = True
        print("[TATR] detection and structure models loaded")

        tables = extractor.extract_tables(
            uploaded,
            page_numbers=_parse_pages(args.pages),
            dpi=args.dpi,
        )
        methods = sorted({table.extraction_method for table in tables})
        report["table_count"] = len(tables)
        report["extraction_methods"] = methods
        report["table_pages"] = [table.page for table in tables]
        report["table_models"] = sorted(
            {
                str(table.raw.get("model"))
                for table in tables
                if table.raw.get("model")
            }
        )

        if any(method != "table_transformer" for method in methods):
            raise AssertionError(
                f"unexpected extraction method(s): {methods}; strict TATR test rejects fallback"
            )
        if args.require_table and not tables:
            raise AssertionError("TATR ran but returned no table; --require-table was specified")

        report["status"] = "passed"
        print(f"[TATR] inference completed; table_count={len(tables)}")
        print(f"[TATR] extraction_methods={methods}")
        print("[TATR] PASS: this run used the project TATR adapter, not pdfplumber fallback")
    except Exception as exc:  # noqa: BLE001 - CLI must report all startup failures
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        print(f"[TATR] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if args.output_json:
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[TATR] report={output_path}")

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
