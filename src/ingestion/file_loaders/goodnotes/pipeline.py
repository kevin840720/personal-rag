from __future__ import annotations

from typing import List, Tuple, Protocol

from PIL import Image

from .types import (
    PageImage,
    DetBox,
    DetectionResult,
    CropItem,
    CropsBatch,
    RecItem,
    RecognitionResult,
    GroupItem,
    GroupedCorpus,
)
from ingestion.file_loaders.goodnotes.crop import CropOps
from ingestion.file_loaders.goodnotes.grouping import GroupingOps
from ingestion.file_loaders.goodnotes.preprocess import GoodNotesPreprocessOps

class TextDetector(Protocol):
    def predict(self, image: Image.Image) -> List[DetBox]:
        ...


class TextRecognizer(Protocol):
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        ...


class GoodnotesOCRPipeline:
    def __init__(self,
                 detector:TextDetector,
                 recognizer:TextRecognizer,
                 ) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def _run_detection(self, page_for_det: PageImage) -> DetectionResult:
        boxes = self.detector.predict(page_for_det.image)
        return DetectionResult(page=page_for_det, boxes=boxes)

    def _run_cropping(self, page_for_crop: PageImage, det: DetectionResult, expand_ratio: float = 1.0) -> CropsBatch:
        return CropOps.make_crops(page_for_crop, det, expand_ratio=expand_ratio)

    def _run_recognition(self, crops: CropsBatch) -> RecognitionResult:
        rec_items: List[RecItem] = []
        for it in crops.items:
            text, score = self.recognizer.predict(it.image)
            rec_items.append(RecItem(crop=it, text=text, score=score))
        return RecognitionResult(items=rec_items)

    def _group_into_corpus(self, rec: RecognitionResult) -> GroupedCorpus:
        return GroupingOps.group_into_corpus(rec)

    # --- public orchestration methods ---
    def run_note_page(self,
                      page:PageImage,
                      ) -> GroupedCorpus:
        """處理黑底筆記頁中的白字

        Args:
            page (PageImage): _description_

        Returns:
            GroupedCorpus: _description_
        """
        page_writing_and_frame, page_writing = GoodNotesPreprocessOps.preprocess_black_notes_text(page)
        det_res = self._run_detection(page_writing_and_frame)
        crops = self._run_cropping(page_writing, det_res, expand_ratio=1.0)
        rec_res = self._run_recognition(crops)
        return self._group_into_corpus(rec_res)

    def run_color_notes(self, page: PageImage) -> GroupedCorpus:
        color_page = GoodNotesPreprocessOps.preprocess_black_notes_handwriting(page)
        det_res = self._run_detection(color_page)
        crops = self._run_cropping(color_page, det_res, expand_ratio=1.0)
        rec_res = self._run_recognition(crops)
        return self._group_into_corpus(rec_res)

    def run_textbook_page(self, page: PageImage) -> GroupedCorpus:
        page_51 = GoodNotesPreprocessOps.preprocess_white_textbook_text(page)
        det_res = self._run_detection(page_51)
        crops = self._run_cropping(page_51, det_res, expand_ratio=1.0)
        rec_res = self._run_recognition(crops)
        return self._group_into_corpus(rec_res)
