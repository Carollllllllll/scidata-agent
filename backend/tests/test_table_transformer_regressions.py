from __future__ import annotations

import subprocess

from PIL import Image

from scidata_agent.agent.schemas import UploadedFile
from scidata_agent.tools import table_transformer


class _FakeCroppedPage:
    def extract_text(self) -> str:
        return "value"


class _FakePage:
    width = 612
    height = 792

    def crop(self, _bbox):
        return _FakeCroppedPage()

    def extract_words(self):
        return []


class _FakePdf:
    def __init__(self) -> None:
        self.pages = [_FakePage()]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def close(self):
        return None


def test_table_extraction_opens_pdf_only_once(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "table.pdf"
    pdf_path.write_bytes(b"placeholder")
    image_path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(image_path)
    open_calls: list[str] = []

    import pdfplumber

    def fake_open(path):
        open_calls.append(str(path))
        return _FakePdf()

    monkeypatch.setattr(pdfplumber, "open", fake_open)
    monkeypatch.setattr(table_transformer, "_render_pages_in_subprocess", lambda *_args: {1: str(image_path)})

    extractor = object.__new__(table_transformer.TableTransformerExtractor)
    extractor._load_models = lambda: None
    extractor._detect_tables = lambda *_args, **_kwargs: [[0, 0, 100, 100]]
    extractor._recognize_structure = lambda *_args, **_kwargs: {
        "rows": [[0, 0, 100, 50], [0, 50, 100, 100]],
        "cols": [[0, 0, 50, 100], [50, 0, 100, 100]],
        "cells": [
            {"row": 0, "col": 0, "bbox": [0, 0, 50, 50]},
            {"row": 0, "col": 1, "bbox": [50, 0, 100, 50]},
            {"row": 1, "col": 0, "bbox": [0, 50, 50, 100]},
            {"row": 1, "col": 1, "bbox": [50, 50, 100, 100]},
        ],
    }

    tables = extractor.extract_tables(UploadedFile(filename="table.pdf", path=pdf_path))

    assert len(tables) == 1
    assert open_calls == [str(pdf_path)]


def test_pdf_renderer_has_a_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SCIDATA_PDF_RENDER_TIMEOUT_SECONDS", "7")

    def time_out(*_args, **kwargs):
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(cmd="renderer", timeout=7)

    monkeypatch.setattr(table_transformer.subprocess, "run", time_out)

    try:
        table_transformer._render_pages_in_subprocess(str(tmp_path / "bad.pdf"), [1], 200)
    except RuntimeError as exc:
        assert "timed out after 7 seconds" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("renderer timeout must be surfaced")
