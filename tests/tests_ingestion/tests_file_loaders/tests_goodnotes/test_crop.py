"""
@File    :  test_crop.py
@Time    :  2025/08/29 15:10:00
@Author  :  Kevin Wang
@Desc    :  Unit tests for CropOps
"""

from __future__ import annotations

from PIL import Image
import numpy as np

from ingestion.file_loaders.goodnotes.types import (PageImage,
                                                    DetBox,
                                                    DetectionResult,
                                                    )
from ingestion.file_loaders.goodnotes.crop import CropOps




class TestCropOps:
    def test_make_crops_expand(self):
        """驗證裁切時的 bbox 擴展與邊界裁切行為。

        - 測試兩個多邊形：一個在中央（可直接擴展），一個在左上角（測試邊界裁切）。
        - 設定 expand_ratio=3.0，檢查最終的 bounding box 及裁切尺寸。
        - 同時確認欄位正確傳遞（poly_origin, det_score, expand_ratio）。
        """

        def make_page(w=100, h=80, color=(255, 255, 255), filename="demo.pdf", page=1) -> PageImage:
            img = Image.new("RGB", (w, h), color=color)
            return PageImage(img, filename, page)

        page = make_page(100, 80)

        # Poly A: centered-ish, bbox (30,30,35,35) -> expand by 3x => (25,25,40,40)
        poly_a = [[30, 30], [35, 30], [35, 35], [30, 35]]
        score_a = 0.95
        # Poly B: near (0,0), bbox (0,0,2,3) -> expand by 3x => (-2,-3,4,6) then clamp => (0,0,4,6)
        poly_b = [[0, 0], [2, 0], [2, 3], [0, 3]]
        score_b = 0.80

        det = DetectionResult(page=page,
                              boxes=[DetBox(poly=poly_a, score=score_a),
                                     DetBox(poly=poly_b, score=score_b),
                                     ])

        crops = CropOps.make_crops(page, det, expand_ratio=3.0)

        assert len(crops.items) == 2

        # Check first crop (centered)
        c0 = crops.items[0]
        assert c0.box == (25, 25, 40, 40)
        assert c0.image.size == (15, 15)  # width,height
        assert c0.poly_origin == poly_a
        assert c0.det_score == score_a
        assert c0.expand_ratio == 3.0

        # Check second crop (near boundary, clamped)
        c1 = crops.items[1]
        assert c1.box == (0, 0, 4, 6)
        assert c1.image.size == (4, 6)
        assert c1.poly_origin == poly_b
        assert c1.det_score == score_b
        assert c1.expand_ratio == 3.0
