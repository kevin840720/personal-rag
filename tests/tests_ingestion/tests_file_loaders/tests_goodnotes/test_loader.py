"""
@File    :  test_loader.py
@Time    :  2025/08/30 00:00:00
@Author  :  Kevin Wang
@Desc    :  Lightweight tests for GoodnotesLoader with stubbed OCR and mocked page images
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np
from PIL import Image
from pypdf import PdfWriter

from ingestion.file_loaders.goodnotes.loader import GoodnotesLoader
from ingestion.file_loaders.goodnotes.types import PageImage, DetBox


class _FakeDetector:
    def predict(self, image: Image.Image):
        w, h = image.size
        return [DetBox(poly=[[10, 10], [w // 2, 10], [w // 2, 30], [10, 30]], score=0.9)]


class _FakeRecognizer:
    def predict(self, image: Image.Image):
        return "DEMO_TEXT", 0.99


def _yield_two_pages(pdf_path: Path) -> Iterator[PageImage]:
    # page 1: bright (white bg), page 2: dark (black bg)
    img1 = Image.fromarray(np.full((20, 30, 3), 240, dtype=np.uint8))
    img2 = Image.fromarray(np.full((20, 30, 3), 10, dtype=np.uint8))
    yield PageImage(img1, pdf_path.name, 1, variant="full")
    yield PageImage(img2, pdf_path.name, 2, variant="full")


class TestGoodnotesLoader:
    def test_load_with_stubs(self, monkeypatch, tmp_path: Path):
        # create a minimal 2-page PDF (no outlines)
        pdf_path = tmp_path / "stub.pdf"
        writer = PdfWriter()
        for _ in range(2):
            writer.add_blank_page(width=200, height=200)
        with open(pdf_path, "wb") as f:
            writer.write(f)

        # mock page images generator to avoid pdf2image/poppler
        from ingestion.file_loaders.goodnotes import ops as _ops_mod
        monkeypatch.setattr(_ops_mod.PdfOps, "pdf_page_images", lambda p, dpi=600: _yield_two_pages(Path(p)))

        loader = GoodnotesLoader(dpi=150, detector=_FakeDetector(), recognizer=_FakeRecognizer())
        results = loader.load(pdf_path)

        assert len(results) == 2
        # content from fake recognizer
        assert all(r.content == "DEMO_TEXT" for r in results)
        # per-page metadata basics
        assert results[0].metadata.page == 1 and results[1].metadata.page == 2
        # background mode detected via brightness
        assert results[0].metadata.extra.get("bg_mode") == "white"
        assert results[1].metadata.extra.get("bg_mode") == "black"
        # outlines empty for this synthetic PDF
        assert results[0].metadata.outlines == []
        assert results[1].metadata.outlines == []

