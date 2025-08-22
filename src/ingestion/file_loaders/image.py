# -*- encoding: utf-8 -*-
"""
@File    :  image.py
@Time    :  2025/08/21 20:00:00
@Author  :  Kevin Wang
@Desc    :  
"""
from pathlib import Path
from typing import (List,
                    Literal,
                    Union,
                    )
from typing import List, Union, Optional

from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from objects import DocumentMetadata, FileType

from datetime import datetime
from docling.document_converter import DocumentConverter

from PIL import Image, ExifTags


class DoclingImageLoader(DocumentLoader):
    """用 Docling 解析影像，產出 LoaderResult
    為了直接套用 Docling 的 OCR 機制，所以用 PDFPipeline 讀取，架構與 DoclingPDFReader 大致相同
    差異：修正 _get_docling_ocr_option 在最後 format_options 的設置
         重寫 _get_metadata

    """

    def __init__(self, do_ocr=False) -> None:
        super().__init__()
        self.converter:DocumentConverter = self._load_converter(do_ocr=do_ocr)

    def _get_docling_ocr_option(self,
                                ocr_model:Literal["PPv4","PPv5"]="PPv5",
                                ):
        """
        建立 Docling Image OCR Option，回傳 RapidOcrOptions 實例。
        支援 PPv4/PPv5，需先下載好 onnx 模型（預設用 PPv5）。
        """
        from huggingface_hub import snapshot_download
        from docling.datamodel.pipeline_options import RapidOcrOptions
        import os

        if ocr_model == "PPv5":
            print("Downloading RapidOCR PPv5 models")
            download_path = snapshot_download(repo_id="SWHL/RapidOCR")
            cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
            ocr_options = RapidOcrOptions(
                lang=["english", "chinese", "japanese"],  # 實測 docling 只認 det/rec 模型
                force_full_page_ocr=False,
                det_model_path="./models/ocr/PP-OCRv5_server_det/inference.onnx",
                rec_model_path="./models/ocr/PP-OCRv5_server_rec/inference.onnx",
                cls_model_path=cls_model_path,
                rec_keys_path="./models/ocr/PP-OCRv5_server_rec/chars.txt"
            )
        elif ocr_model == "PPv4":
            print("Downloading RapidOCR PPv4 models")
            download_path = snapshot_download(repo_id="SWHL/RapidOCR")
            det_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_det_infer.onnx")
            rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_infer.onnx")
            cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
            ocr_options = RapidOcrOptions(
                lang=["english", "chinese", "japanese"],
                force_full_page_ocr=False,
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
            )
        else:
            raise AttributeError("Unsupported Model")
        return ocr_options

    def _load_converter(self, do_ocr:bool=True) -> DocumentConverter:
        """
        回傳配置好 OCR 及自訂 image pipeline 的 Docling DocumentConverter
        """
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend

        pipeline_options = PdfPipelineOptions()  # Hum... 我看網路上會直接用 PDFPipeline，Docling 對 Image 支援流於表面而已

        if do_ocr:
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = self._get_docling_ocr_option()
        else:
            pipeline_options.do_ocr = False

        # Image 應該是沒有 Table 的架構啦
        pipeline_options.do_table_structure = False
        pipeline_options.table_structure_options.do_cell_matching = False   

        format_options = {
            InputFormat.IMAGE: FormatOption(pipeline_cls=StandardPdfPipeline,
                                           pipeline_options=pipeline_options,
                                           backend=DoclingParseV4DocumentBackend,
                                           )
            
        }

        return DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options=format_options
        )

    def _get_exif_dt(self,
                     path:Path,
                     ) -> Optional[datetime]:
        # TODO: 未檢查，此函數是獲取「拍攝時間」用的
        if Image is None:
            return None
        try:
            with Image.open(str(path)) as im:
                exif = getattr(im, "_getexif", None)
                if exif is None:
                    return None
                data = exif() or {}
                # 找出 DateTimeOriginal 的 key
                dt_key = None
                if ExifTags is not None:
                    for k, v in ExifTags.TAGS.items():
                        if v == "DateTimeOriginal":
                            dt_key = k
                            break
                raw = data.get(dt_key) if dt_key is not None else None
                if isinstance(raw, str):
                    # 典型格式 "YYYY:MM:DD HH:MM:SS"
                    raw = raw.replace("/", ":")
                    try:
                        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        return None
        except Exception:
            return None
        return None

    def _get_metadata(self,
                      path:Union[str,Path],
                      ) -> DocumentMetadata:
        file_path = Path(path)
        st = file_path.stat()
        created_at = self._get_exif_dt(file_path) or datetime.fromtimestamp(st.st_ctime)
        modified_at = datetime.fromtimestamp(st.st_mtime)
        return DocumentMetadata(file_type=FileType.IMAGE,
                                file_name=file_path.name,
                                created_at=created_at,
                                modified_at=modified_at,
                                )

    def load(self,
             path:Union[str, Path],
             ) -> List[LoaderResult]:
        file_path = Path(path)
        docling_doc = self.converter.convert(str(file_path)).document
        metadata = self._get_metadata(file_path)
        content = export_to_user_text(docling_doc)
        return [LoaderResult(content=content, metadata=metadata, doc=docling_doc)]
