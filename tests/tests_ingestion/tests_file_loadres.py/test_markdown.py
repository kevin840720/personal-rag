# -*- encoding: utf-8 -*-
"""
@File    :  test_markdown.py
@Time    :  2025/08/16 16:43:46
@Author  :  Kevin Wang
@Desc    :  None
"""

from datetime import datetime
from pathlib import Path
from typing import List

from docling_core.types.doc.document import DoclingDocument
import pytest

from ingestion.base import LoaderResult
from ingestion.file_loaders.markdown import DoclingMarkdownLoader
from objects import FileType

@pytest.fixture
def test_md_path(tmp_path:Path) -> Path: # tmp_path 是 pytest 內建 fixture，會提供臨時目錄並於測試後自動清理
    """建立一個臨時的 markdown 檔案，供單一檔案測試使用"""
    file_path = tmp_path.joinpath("test01.md")
    file_path.write_text("# 測試文件\n\n臨時 markdown 檔案。",
                         encoding="utf-8",
                         )
    return file_path

@pytest.fixture
def test_md_paths(tmp_path:Path) -> List[Path]:
    """建立多個臨時的 markdown 檔案 (pytest 測試結束後會自動刪除)"""
    file1 = tmp_path.joinpath("test01.md")  # tmp_path 與 fixture name 有關，都取名 file01 不會衝突
    file2 = tmp_path.joinpath("test02.md")

    file1.write_text("# 測試文件 01\n\n內容一。", encoding="utf-8")
    file2.write_text("# 測試文件 02\n\n內容二。", encoding="utf-8")

    return [file1, file2]

class TestDoclingMarkdownLoader:
    @pytest.fixture
    def markdown_loader(self):
        return DoclingMarkdownLoader()

    def test_load(self,
                  markdown_loader,
                  test_md_path:Path,
                  ):
        results = markdown_loader.load(test_md_path)
        assert isinstance(results, list) and len(results) == 1

        res = results[0]
        assert isinstance(res, LoaderResult)

        # metadata 檢查
        assert res.metadata.file_type == FileType.MARKDOWN
        assert res.metadata.file_name == test_md_path.name
        assert isinstance(res.metadata.created_at, datetime)
        assert isinstance(res.metadata.modified_at, datetime)

        # content 與 doc 基本檢查（只要能包含原文關鍵字即可）
        assert isinstance(res.content, str) and "測試文件" in res.content
        assert isinstance(res.doc, DoclingDocument)

    def test_load_multi(self,
                        markdown_loader,
                        test_md_paths,
                        ):
        results = markdown_loader.load_multi(test_md_paths)

        assert isinstance(results, list)
        assert len(results) == len(test_md_paths)

        for res, src_path in zip(results, test_md_paths):
            assert isinstance(res, LoaderResult)

            # metadata 檢查
            assert res.metadata.file_type == FileType.MARKDOWN
            assert res.metadata.file_name == src_path.name
            assert isinstance(res.metadata.created_at, datetime)
            assert isinstance(res.metadata.modified_at, datetime)

            # content 與 doc 基本檢查
            assert isinstance(res.content, str) and "測試" in res.content
            assert isinstance(res.doc, DoclingDocument)
