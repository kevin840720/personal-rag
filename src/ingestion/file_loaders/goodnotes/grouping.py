from __future__ import annotations

from typing import (Dict,
                    List,
                    Tuple,
                    )

from ingestion.file_loaders.goodnotes.ops import (Clustering,
                                                  Geometry,
                                                  )
from ingestion.file_loaders.goodnotes.types import (RecognitionResult,
                                                    GroupItem,
                                                    GroupedCorpus,
                                                    )


class GroupingOps:
    """將辨識結果依幾何關係分群，再彙整為語料。

    - 以 bbox 重疊為基準進行群組化（可調整擴張比例）；
    - 依視覺閱讀順序（先上後下、同列從左到右）排序群內項目；
    - 將各群文字串接，輸出聚合文本與幾何資訊。
    """

    @classmethod
    def group_into_corpus(cls,
                          rec:RecognitionResult,
                          ) -> GroupedCorpus:
        """把辨識結果整理成 `GroupedCorpus`。

        Args:
            rec (RecognitionResult): 文字辨識的批次輸出（含每個裁切與其文字）。

        Returns:
            GroupedCorpus: 聚合後的語料（含群組與內容）。
        """
        items: List[Dict] = []
        for r in rec.items:
            x0, y0, x1, y1 = r.crop.box
            x_center = (x0 + x1) / 2
            y_center = (y0 + y1) / 2
            items.append({"rec_text": r.text,
                          "x_min": x0,
                          "x_max": x1,
                          "y_min": y0,
                          "y_max": y1,
                          "x_center": x_center,
                          "y_center": y_center,
                          "rectangle": Geometry.bbox_to_poly((x0, y0, x1, y1)),
                          })

        # Clustering.overlap 接受 bbox tuple，並回傳索引群組
        bboxes: List[Tuple[float, float, float, float]] = [
            (d["x_min"], d["y_min"], d["x_max"], d["y_max"]) for d in items
        ]
        groups_idx = Clustering.overlap(bboxes, height_ratio=2.0, width_ratio=None)

        def group_key(idxs: List[int]) -> Tuple[float, float]:
            min_y = min(items[i]["y_center"] for i in idxs)
            min_x = min(items[i]["x_center"] for i in idxs if items[i]["y_center"] == min_y)
            return (min_y, min_x)

        groups_sorted = sorted(groups_idx, key=group_key)
        result_groups: List[GroupItem] = []
        all_contents: List[str] = []
        for g_idxs in groups_sorted:
            g_sorted_idxs = sorted(g_idxs, key=lambda i: (items[i]["y_center"], items[i]["x_center"]))
            rects = [items[i]["rectangle"] for i in g_sorted_idxs]
            texts = [items[i]["rec_text"] for i in g_sorted_idxs]
            all_contents.extend(texts)
            result_groups.append(
                GroupItem(box=items[g_sorted_idxs[0]]["rectangle"],
                          contents=texts,
                          rectangles={"counts": len(g_sorted_idxs), "items": rects},
                          )
            )

        content = "\n".join(all_contents)
        return GroupedCorpus(content=content, group=result_groups)
