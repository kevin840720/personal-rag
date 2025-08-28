from __future__ import annotations

from typing import List

from ingestion.file_loaders.goodnotes.ops import Geometry
from ingestion.file_loaders.goodnotes.types import (PageImage,
                                                    DetectionResult,
                                                    CropItem,
                                                    CropsBatch,
                                                    )

class CropOps:
    """裁切相關操作。
    """

    @classmethod
    def make_crops(cls,
                   page_for_crop:PageImage,
                   det:DetectionResult,
                   expand_ratio:float=1.0,
                   ) -> CropsBatch:
        """根據偵測結果裁切頁面並回傳批次結果。

        Args:
            page_for_crop (PageImage): 來源頁面（含 PIL 圖與識別資訊）。
            det (DetectionResult): 偵測輸出，包含多個 polygon 與信心分數。
            expand_ratio (float, optional): 寬、高各自按比例外擴的倍數（=1 表示不擴）。

        Returns:
            CropsBatch: 含多個 `CropItem` 的集合。
        """
        W, H = page_for_crop.size
        items: List[CropItem] = []
        for db in det.boxes:
            x_min, y_min, x_max, y_max = Geometry.poly_to_bbox(db.poly)
            ex_x0, ex_y0, ex_x1, ex_y1 = Geometry.expand_bbox((x_min, y_min, x_max, y_max),
                                                               height_ratio=expand_ratio,
                                                               width_ratio=expand_ratio,
                                                               img_size=(W, H),
                                                               )
            crop_img = page_for_crop.image.crop((ex_x0, ex_y0, ex_x1, ex_y1))
            items.append(
                CropItem(page=page_for_crop,
                         box=(ex_x0, ex_y0, ex_x1, ex_y1),
                         poly_origin=db.poly,
                         det_score=db.score,
                         image=crop_img,
                         expand_ratio=expand_ratio,
                         )
            )
        return CropsBatch(items=items)
