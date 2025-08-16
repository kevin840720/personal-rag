# -*- encoding: utf-8 -*-
"""
@File    :  docx.py
@Time    :  2025/01/23 11:37:14
@Author  :  Kevin Wang
@Desc    :  Docx 的 oxml 格式太過魔幻，所以直接交給 Docling 處理
"""

from pathlib import Path
from typing import List, Union
from docx import Document as DocxDocument

from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from objects import DocumentMetadata, FileType

from docling.document_converter import DocumentConverter

class DoclingDocxLoader(DocumentLoader):
    """用 Docling 解析 DOCX，產出 LoaderResult"""

    def __init__(self):
        super().__init__()
        self.converter = DocumentConverter()

    def _get_metadata(self, path: Union[Path, str]) -> DocumentMetadata:
        doc = DocxDocument(str(path))
        props = doc.core_properties
        return DocumentMetadata(file_type=FileType.DOCX,
                                file_name=Path(path).name,
                                title=props.title,
                                author=props.author,
                                subject=props.subject,
                                created_at=props.created,
                                modified_at=props.modified,
                                )

    def load(self,
             path:Union[str,Path],
             ) -> List[LoaderResult]:
        path = Path(path)
        docling_doc = self.converter.convert(str(path)).document
        metadata = self._get_metadata(path)
        content = export_to_user_text(docling_doc)
        result = LoaderResult(content=content,
                              metadata=metadata,
                              doc=docling_doc,
                              )
        return [result]
