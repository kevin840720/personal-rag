from __future__ import annotations

from typing import Dict, List, Optional, Union
from pathlib import Path

from pypdf import PdfReader
from PIL import Image

from ingestion.base import DocumentLoader, LoaderResult
from objects import PDFMetadata, DocumentMetadata, FileType

from .types import PageImage, GroupedCorpus
from .ops import PdfOps, ImageOps
from .pipeline import GoodnotesOCRPipeline


class GoodnotesMetadata(PDFMetadata):
    page: int
    outlines: List[str]

    def to_dict(self):
        result = super().to_dict()
        result["page"] = self.page
        result["outlines"] = self.outlines
        return result

    @classmethod
    def from_dict(cls, data: Dict):
        base = PDFMetadata.from_dict(data)
        return cls(
            file_type=base.file_type,
            file_name=base.file_name,
            title=base.title,
            author=base.author,
            subject=base.subject,
            created_at=base.created_at,
            modified_at=base.modified_at,
            source=base.source,
            producer=base.producer,
            page=data.get("page", 1),
            outlines=data.get("outlines", []),
        )


def _flatten_outlines(reader: PdfReader) -> Dict[int, List[str]]:
    """Builds page→[titles] map from a PDF's outlines (bookmarks).

    GoodNotes commonly uses a single-level outline. This function is robust to nested lists.
    """
    page_map: Dict[int, List[str]] = {}

    def handle_item(item):
        try:
            title = getattr(item, "title", None)
            if title is None:
                return
            page_num = reader.get_destination_page_number(item) + 1
            page_map.setdefault(page_num, []).append(str(title))
        except Exception:
            return

    try:
        outlines = getattr(reader, "outline", None)
        if outlines is None:
            outlines = getattr(reader, "outlines", None)
    except Exception:
        outlines = None

    def walk(node):
        if node is None:
            return
        if isinstance(node, list):
            for n in node:
                walk(n)
        else:
            handle_item(node)

    walk(outlines)
    return page_map


class _PaddleTextDetector:
    def __init__(self, model_name: str = "PP-OCRv5_server_det", batch_size: int = 1):
        from paddleocr import TextDetection  # type: ignore

        self.model = TextDetection(model_name=model_name)
        self.batch_size = batch_size

    def predict(self, image: Image.Image):
        import numpy as np

        from ingestion.file_loaders.goodnotes.types import DetBox

        np_img = np.array(image.convert("RGB"))
        output = self.model.predict(np_img, batch_size=self.batch_size)
        boxes: List[DetBox] = []
        for res in output:
            data = getattr(res, "json", None)
            if isinstance(data, dict):
                data = data.get("res", data)
            else:
                try:
                    data = res["res"]
                except Exception:
                    data = None
            if not isinstance(data, dict):
                continue
            polys = data.get("dt_polys", [])
            scores = data.get("dt_scores", [])
            for poly, score in zip(polys, scores):
                poly_list = [[int(p[0]), int(p[1])] for p in poly]
                boxes.append(DetBox(poly=poly_list, score=float(score)))
        return boxes

class _PaddleTextRecognizer:
    def __init__(self, model_name: str = "PP-OCRv5_server_rec", batch_size: int = 1):
        from paddleocr import TextRecognition  # type: ignore

        self.model = TextRecognition(model_name=model_name)
        self.batch_size = batch_size

    def predict(self, image: Image.Image):
        import numpy as np

        np_img = np.array(image.convert("RGB"))
        output = self.model.predict(np_img, batch_size=self.batch_size)
        if not output:
            return "", 0.0
        res = output[0]
        data = getattr(res, "json", None)
        if isinstance(data, dict):
            payload = data.get("res", data)
        else:
            try:
                payload = res["res"]  # type: ignore[index]
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            return "", 0.0
        text = payload.get("rec_text", "")
        score = float(payload.get("rec_score", 0.0))
        return text, score


class GoodnotesLoader(DocumentLoader):
    """Load GoodNotes-exported PDFs via OCR pipeline.

    For each page:
      1) Estimate background (white/black)
      2) Run OCR pipeline variants to produce trunk (主幹) and note (筆記)
      3) Return LoaderResult per page: content = trunk, metadata.extra.note = note
    """

    def __init__(
        self,
        dpi:int=1000,
        detector: Optional[object] = None,
        recognizer: Optional[object] = None,
    ):
        super().__init__()
        self.dpi = dpi
        self.detector = detector or _PaddleTextDetector()
        self.recognizer = recognizer or _PaddleTextRecognizer()
        self.pipeline = GoodnotesOCRPipeline(detector=self.detector, recognizer=self.recognizer)

    def _get_metadata(self, path: Union[str, Path]) -> DocumentMetadata:
        reader = PdfReader(path)
        props = reader.metadata
        if props:
            meta = PDFMetadata(file_type=FileType.PDF,
                               file_name=Path(path).name,
                               title=props.title,
                               author=props.creator,
                               subject=props.subject,
                               created_at=props.creation_date,
                               modified_at=props.modification_date,
                               producer=props.producer,
                               )
        else:
            meta = PDFMetadata(file_type=FileType.PDF,
                               file_name=Path(path).name,
                               )
        return meta

    def _get_page_metadatas(self,
                            path:Union[str,Path],
                            ) -> Dict[int,GoodnotesMetadata]:
        """_summary_

        Args:
            path (Union[str,Path]): _description_

        Returns:
            Dict[int,GoodnotesMetadata]: _description_
        """
        reader = PdfReader(path)
        base_meta = self._get_metadata(path).to_dict()
        page_outline_map = _flatten_outlines(reader)
        pages = len(reader.pages)
        result: Dict[int, GoodnotesMetadata] = {}
        for page_num in range(1, pages + 1):
            outlines = page_outline_map.get(page_num, [])
            gm = GoodnotesMetadata(**base_meta,
                                   page=page_num,
                                   outlines=outlines,
                                   )
            result[page_num] = gm
        return result

    def _run_page(self,
                  page:PageImage,
                  is_white_bg:bool,
                  ) -> tuple[GroupedCorpus, GroupedCorpus]:
        """_summary_ (黑底還是白底)

        Args:
            page (PageImage): _description_
            is_white_bg (bool): _description_

        Returns:
            tuple[GroupedCorpus, GroupedCorpus]: _description_
        """
        if is_white_bg:
            trunk = self.pipeline.run_textbook_page(page)
            note = self.pipeline.run_color_notes(page)
        else:
            trunk = self.pipeline.run_note_page(page)
            note = self.pipeline.run_color_notes(page)
        return trunk, note

    def load(self,
             path:Union[str,Path],
             ) -> List[LoaderResult]:
        """_summary_

        Args:
            path (Union[str,Path]): _description_

        Returns:
            List[LoaderResult]: _description_
        """
        path = Path(path)
        page_metas = self._get_page_metadatas(path)

        results:List[LoaderResult] = []
        for idx, page in enumerate(PdfOps.pdf_page_images(path, dpi=self.dpi), start=1):
            # 1) 區分筆記是白底還是黑底
            bg = ImageOps.estimate_background_color(page.image)
            is_white = (bg == "white")

            # 2) run OCR variants
            trunk, note = self._run_page(page, is_white_bg=is_white)

            # 3) build metadata per page
            meta = page_metas.get(idx)
            if meta is None:
                # fallback if mismatch
                base = self._get_metadata(path)
                meta = GoodnotesMetadata(file_type=base.file_type,
                                         file_name=base.file_name,
                                         title=base.title,
                                         author=base.author,
                                         subject=base.subject,
                                         created_at=base.created_at,
                                         modified_at=base.modified_at,
                                         source=base.source,
                                         producer=base.producer,
                                         page=idx,
                                         outlines=[],
                                         )

            # pack extra fields
            extra = meta.extra.copy() if meta.extra else {}
            extra.update({"bg_mode": "white" if is_white else "black"})
            meta.extra = extra

            # 4) LoaderResult: content=trunk, doc=None
            results.append(LoaderResult(content=trunk.content, metadata=meta, doc=None))

        return results
