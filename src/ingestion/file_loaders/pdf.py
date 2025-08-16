# -*- encoding: utf-8 -*-
"""
@File    :  pdf.py
@Time    :  2025/01/21 16:06:08
@Author  :  Kevin Wang
@Desc    :  用 Docling 處理 PDF，產出 LoaderResult
"""

from enum import Enu
from typing import (List,
                    Union,
                    )
from pathlib import Path
import os
import re
import uuid

from pypdf import PdfReader
from docling.document_converter import DocumentConverter

from pypdf import PdfReader
from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from objects import (
                     DocumentMetadata,
                     FileType,
                     PDFSourceType,
                     PDFMetadata,
                     )



from pathlib import Path
from typing import List, Union

from docling.document_converter import DocumentConverter
from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from objects import (DocumentMetadata,
                     FileType,
                     PDFSourceType,
                     PDFMetadata,
                     )
from pypdf import PdfReader

class DoclingPDFLoader(DocumentLoader):
    def __init__(self):
        super().__init__()
        self.converter: DocumentConverter = self._load_converter()
        self._pattern = {
            PDFSourceType.MS_EXCEL: re.compile(r"Microsoft.+Excel"),
            PDFSourceType.MS_WORD: re.compile(r"Microsoft.+Word"),
            PDFSourceType.PDF: re.compile(r"Adobe.+PDF"),
            PDFSourceType.LO_WRITER: re.compile(r'(?=.*LibreOffice)(?=.*Writer)', re.IGNORECASE),
            PDFSourceType.LO_CALC: re.compile(r'(?=.*LibreOffice)(?=.*Calc)', re.IGNORECASE),
        }

    def _get_docling_ocr_option(self):
        from huggingface_hub import snapshot_download
        from docling.datamodel.pipeline_options import RapidOcrOptions  # Docling 中，唯一支援中文
        
        # Download RappidOCR models from HuggingFace
        print("Downloading RapidOCR models")
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")
        
        # Setup RapidOcrOptions for english detection
        # PP-OCRv5 已經在 2025.06.05 釋出，近期可以多關注是否被轉移到 HuggingFace
        det_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_det_infer.onnx")
        rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_infer.onnx")
        cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
        ocr_options = RapidOcrOptions(lang=['chinese'],
                                      force_full_page_ocr=False,
                                      det_model_path=det_model_path,
                                      rec_model_path=rec_model_path,
                                      cls_model_path=cls_model_path,
                                      )
        return ocr_options

    def _load_converter(self) -> DocumentConverter:
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                        TableFormerMode,
                                                        )
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Force full page OCR
        pipeline_options.ocr_options = self._get_docling_ocr_option()
        pipeline_options.do_table_structure = True  # Same as default
        pipeline_options.table_structure_options.do_cell_matching = True  # Same as default
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # Same as default

        format_options = {
            InputFormat.PDF: FormatOption(pipeline_cls=StandardPdfPipeline,
                                          pipeline_options=pipeline_options,
                                          backend=DoclingParseV4DocumentBackend,
                                          )
        }

        return DocumentConverter(format_options=format_options)

    def _check_pdf_source(self, path: Union[str, Path]) -> PDFSourceType:
        metadata = PdfReader(path).metadata
        creator = (metadata.get('/Creation', '') or '') + (metadata.get('/Producer', '') or '')
        for creator_type, pattern in self._pattern.items():
            if pattern.search(creator):
                return creator_type
        return PDFSourceType.UNKNOWN

    def _get_metadata(self, path: Union[Path, str]) -> DocumentMetadata:
        props = PdfReader(path).metadata
        if props:
            metadata = PDFMetadata(
                file_type=FileType.PDF,
                file_name=Path(path).name,
                title=props.title,
                author=props.creator,
                subject=props.subject,
                created_at=props.creation_date,
                modified_at=props.modification_date,
                source=self._check_pdf_source(path),
                producer=props.producer,
            )
            return metadata
        return PDFMetadata(
            file_type=FileType.PDF,
            file_name=Path(path).name,
        )

    def load(self, path: Union[str, Path]) -> List[LoaderResult]:
        path = Path(path)
        docling_doc = self.converter.convert(str(path)).document
        content = export_to_user_text(docling_doc)
        metadata = self._get_metadata(path)
        result = LoaderResult(content=content,
                              metadata=metadata,
                              doc=docling_doc,
                              )
        return [result]
