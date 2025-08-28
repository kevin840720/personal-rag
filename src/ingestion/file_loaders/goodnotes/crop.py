from __future__ import annotations

from typing import List

from .types import PageImage, DetectionResult, CropItem, CropsBatch
from .ops import Geometry


def make_crops(page_for_crop: PageImage, det: DetectionResult, expand_ratio: float = 1.0) -> CropsBatch:
    W, H = page_for_crop.size
    items: List[CropItem] = []
    for db in det.boxes:
        x_min, y_min, x_max, y_max = Geometry.poly_to_bbox(db.poly)
        w = x_max - x_min
        h = y_max - y_min
        ex_x0, ex_x1 = Geometry._expand_range(x_min, x_max, w, ratio=expand_ratio, pixel=None)  # type: ignore[attr-defined]
        ex_y0, ex_y1 = Geometry._expand_range(y_min, y_max, h, ratio=expand_ratio, pixel=None)  # type: ignore[attr-defined]
        ex_x0 = max(0, ex_x0)
        ex_y0 = max(0, ex_y0)
        ex_x1 = min(W, ex_x1)
        ex_y1 = min(H, ex_y1)
        crop_img = page_for_crop.image.crop((ex_x0, ex_y0, ex_x1, ex_y1))
        items.append(
            CropItem(
                page=page_for_crop,
                box=(ex_x0, ex_y0, ex_x1, ex_y1),
                poly_origin=db.poly,
                det_score=db.score,
                image=crop_img,
                expand_ratio=expand_ratio,
            )
        )
    return CropsBatch(items=items)

