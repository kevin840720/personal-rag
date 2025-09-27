# -*- encoding: utf-8 -*-
"""
@File    :  pdf.py
@Time    :  2025/01/21 16:06:08
@Author  :  Kevin Wang
@Desc    :  用 Docling 處理 PDF，產出 LoaderResult
"""

# from enum import Enu
from typing import (List,
                    Literal,
                    Union,
                    )
from pathlib import Path
import os
import re

from pypdf import PdfReader
from docling.document_converter import DocumentConverter

from pypdf import PdfReader
from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from objects import (DocumentMetadata,
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
    def __init__(self,
                 do_ocr:bool=True,
                 do_table_structure:bool=True,
                 ):
        super().__init__()
        self.converter: DocumentConverter = self._load_converter(do_ocr=do_ocr,
                                                                 do_table_structure=do_table_structure,
                                                                 )
        self._pattern = {
            PDFSourceType.MS_EXCEL: re.compile(r"Microsoft.+Excel"),
            PDFSourceType.MS_WORD: re.compile(r"Microsoft.+Word"),
            PDFSourceType.PDF: re.compile(r"Adobe.+PDF"),
            PDFSourceType.LO_WRITER: re.compile(r'(?=.*LibreOffice)(?=.*Writer)', re.IGNORECASE),
            PDFSourceType.LO_CALC: re.compile(r'(?=.*LibreOffice)(?=.*Calc)', re.IGNORECASE),
        }

    def _get_docling_ocr_option(self, ocr_model:Literal["PPv4", "PPv5"]="PPv5"):
        from huggingface_hub import snapshot_download
        from docling.datamodel.pipeline_options import RapidOcrOptions  # Docling 中，唯一支援中文
        
        if ocr_model == "PPv5":
            # RapidOCR 官方還未支援 PPv5：必須手動下載 PPv5 模型與字典，轉成 onnx。
            print(os.getcwd())
            print("Downloading RapidOCR models")
            det_model_path = "./models/ocr/PP-OCRv5_server_det/inference.onnx"
            rec_model_path = "./models/ocr/PP-OCRv5_server_rec/inference.onnx"
            rec_keys_path = "./models/ocr/PP-OCRv5_server_rec/chars.txt"
            download_path = snapshot_download(repo_id="SWHL/RapidOCR")
            cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
            ocr_options = RapidOcrOptions(lang=["english", "chinese", "japanese"],  # Docling 說這參數沒用
                                          force_full_page_ocr=False,
                                          det_model_path=det_model_path,
                                          rec_model_path=rec_model_path,
                                          cls_model_path=cls_model_path,
                                          rec_keys_path=rec_keys_path,
                                          # 有關 Docling v2.54.0 中，omegaconf.errors.ConfigKeyError: Missing key dict_url 的錯誤
                                          # 原因如下：
                                          # 1. 雖然提供了 rec_keys_path，但在 RapidOCR 內部被映射為 keys_path，get_character_dict() 找不到 rec_keys_path 而走下載路徑。
                                          #    (參考 rapidocr.ch_ppocr_rec.main 中的 TextRecognizer，line 60)
                                          # 2. 手動在 rapidocr_params 中指定 "Rec.rec_keys_path" 可以修復此問題。
                                          # 3. 類似問題已在 GitHub 回報：[docling-project/docling#2249](https://github.com/docling-project/docling/discussions/2249)
                                          rapidocr_params={"Rec.rec_keys_path": rec_keys_path},
                                          )
        elif ocr_model == "PPv4":
            # Download RappidOCR models from HuggingFace
            print("Downloading RapidOCR models")
            download_path = snapshot_download(repo_id="SWHL/RapidOCR")
            
            # Setup RapidOcrOptions for english detection: https://zhuanlan.zhihu.com/p/28420781780
            # PP-OCRv5 已經在 2025.06.05 釋出，近期可以多關注是否被轉移到 HuggingFace
            det_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_det_infer.onnx")
            rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_infer.onnx")
            cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
            ocr_options = RapidOcrOptions(lang=["english", "chinese", "japanese"],  # Docling 說這參數沒用
                                          force_full_page_ocr=False,
                                          det_model_path=det_model_path,
                                          rec_model_path=rec_model_path,
                                          cls_model_path=cls_model_path,
                                          )
        else:
            raise AttributeError("Unsupported Model")
        return ocr_options

    def _load_converter(self,
                        do_ocr:bool=True,
                        do_table_structure:bool=True,
                        ) -> DocumentConverter:
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
        from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                        TableFormerMode,
                                                        )
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        pipeline_options = PdfPipelineOptions()
        if do_ocr:
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = self._get_docling_ocr_option()
        else:
            pipeline_options.do_ocr = False

        if do_table_structure:
            pipeline_options.do_table_structure = True  # Same as default
            pipeline_options.table_structure_options.do_cell_matching = True  # Same as default
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # Same as default
        else:
            pipeline_options.do_table_structure = False
            pipeline_options.table_structure_options.do_cell_matching = False   

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
