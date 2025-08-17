# -*- encoding: utf-8 -*-
"""
@File    :  test_pdf.py
@Time    :  2025/08/16 19:50:00
@Author  :  Kevin Wang
@Desc    :  Tests for DoclingPDFLoader (returns LoaderResult)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest
from pypdf import PdfWriter
from docling_core.types.doc.document import DoclingDocument

from conftest import SKIP_REAL_FILE_TEST
from ingestion.base import LoaderResult
from ingestion.file_loaders.pdf import DoclingPDFLoader
from objects import FileType


@pytest.fixture
def test_pdf_path(tmp_path: Path) -> Path:
    """建立一個臨時的 PDF 檔案並寫入 metadata，供單一檔案測試使用。"""
    file_path = tmp_path / "test01.pdf"
    now = datetime.now(timezone.utc)

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({
        "/Title": "單檔測試標題",
        "/Author": "Kevin Wang",
        "/Subject": "單檔測試主題",
        "/CreationDate": now.strftime("D:%Y%m%d%H%M%S+00'00"),
        "/ModDate": now.strftime("D:%Y%m%d%H%M%S+00'00"),
        "/Producer": "pypdf",
        "/Creator": "pypdf",
    })
    with open(file_path, "wb") as f:
        writer.write(f)
    return file_path


@pytest.fixture
def test_pdf_paths(tmp_path: Path) -> List[Path]:
    """建立多個臨時的 PDF 檔案"""
    paths: List[Path] = []
    now = datetime.now(timezone.utc)

    for i in range(1, 3):
        p = tmp_path / f"test{i:02d}.pdf"
        w = PdfWriter()
        w.add_blank_page(width=612, height=792)
        w.add_metadata({
            "/Title": f"批次檔案{i:02d}",
            "/Author": "Kevin Wang",
            "/Subject": "批次測試",
            "/CreationDate": now.strftime("D:%Y%m%d%H%M%S+00'00"),
            "/ModDate": now.strftime("D:%Y%m%d%H%M%S+00'00"),
            "/Producer": "pypdf",
            "/Creator": "pypdf",
        })
        with open(p, "wb") as f:
            w.write(f)
        paths.append(p)
    return paths


class TestDoclingPDFLoader:
    @pytest.fixture
    def pdf_loader(self):
        return DoclingPDFLoader()

    def test_get_metadata(self,
                          pdf_loader,
                          test_pdf_path:Path,
                          ):
        md = pdf_loader._get_metadata(test_pdf_path)
        assert md.file_type == FileType.PDF
        assert md.file_name == test_pdf_path.name
        if md.created_at is not None:
            assert isinstance(md.created_at, datetime)
        if md.modified_at is not None:
            assert isinstance(md.modified_at, datetime)

    def test_load(self,
                  pdf_loader,
                  test_pdf_path:Path,
                  ):
        results = pdf_loader.load(test_pdf_path)
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        assert isinstance(res, LoaderResult)

        # 檢驗 metadata
        assert res.metadata.file_type == FileType.PDF
        assert res.metadata.file_name == test_pdf_path.name
        assert isinstance(res.metadata.created_at, datetime)
        assert isinstance(res.metadata.modified_at, datetime)

        # content 與 doc 基本檢查（包含原文關鍵字）
        assert isinstance(res.content, str)
        assert isinstance(res.doc, DoclingDocument)

    def test_load_multi(self, pdf_loader, test_pdf_paths: List[Path]):
        results = pdf_loader.load_multi(test_pdf_paths)
        assert isinstance(results, list)
        assert len(results) == len(test_pdf_paths)
        for res, src_path in zip(results, test_pdf_paths):
            assert isinstance(res, LoaderResult)
            assert res.metadata.file_type == FileType.PDF
            assert res.metadata.file_name == src_path.name
            assert isinstance(res.content, str)
            assert isinstance(res.doc, DoclingDocument)

    # @pytest.mark.skipif(SKIP_REAL_FILE_TEST, reason="Skipping real file tests")  # TODO: 挑選合適的 PDF
    # def test_load_real(self, pdf_loader):
    #     file_path = Path("./tests/data/RAG測試文件.pdf")
    #     if not file_path.exists():
    #         pytest.skip("實際檔案不存在，略過此測試")
    #     results = pdf_loader.load(file_path)
    #     assert isinstance(results, list) and len(results) == 1
    #     res = results[0]
    #     assert isinstance(res, LoaderResult)
    #     assert res.metadata.file_type == FileType.PDF
    #     assert res.metadata.file_name == file_path.name
    #     assert isinstance(res.content, str)
    #     assert isinstance(res.doc, DoclingDocument)
