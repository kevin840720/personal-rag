# -*- encoding: utf-8 -*-
"""
@File    :  markdown.py
@Time    :  2025/08/15 06:16:35
@Author  :  Kevin Wang
@Desc    :  用 Docling 解析 Markdown，產出 LoaderResult
"""

from pathlib import Path
from typing import List, Union

from ingestion.base import (DocumentLoader,
                            LoaderResult,
                            )
from ingestion.utils import export_to_user_text
from ingestion.base import LoaderResult  # 加入 LoaderResult
from objects import DocumentMetadata, FileType

from datetime import datetime
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

class DoclingMarkdownLoader(DocumentLoader):
    """Use docling to process Markdown files into LoaderResult objects"""
    def __init__(self):
        super().__init__()
        self.converter = DocumentConverter()

    def _get_metadata(self,
                      path:Union[str,Path],
                      ) -> DocumentMetadata:
        path = Path(path)
        stats = path.stat()
        created_at = datetime.fromtimestamp(stats.st_ctime)
        modified_at = datetime.fromtimestamp(stats.st_mtime)
        return DocumentMetadata(file_type=FileType.MARKDOWN,
                                file_name=path.name,
                                created_at=created_at,
                                modified_at=modified_at,
                                )

    def load(self,
             path:Union[str,Path],
             ) -> List[LoaderResult]:
        path = Path(path)
        docling_doc = self.converter.convert(str(path)).document
        metadata = self._get_metadata(path)
        result = LoaderResult(content=export_to_user_text(docling_doc),
                              metadata=metadata,
                              doc=docling_doc,
                              )
        return [result]
