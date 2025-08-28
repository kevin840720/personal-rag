from __future__ import annotations

from typing import List, Dict, Tuple

from .types import RecognitionResult, GroupItem, GroupedCorpus
from .ops import Clustering, Geometry


def group_into_corpus(rec: RecognitionResult) -> GroupedCorpus:
    items: List[Dict] = []
    for r in rec.items:
        x0, y0, x1, y1 = r.crop.box
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2
        items.append(
            {
                "img": f"{r.crop.page.filename}-p{r.crop.page.page}",  # TODO: 似乎沒使用？
                "rec_text": r.text,
                "x_min": x0,
                "x_max": x1,
                "y_min": y0,
                "y_max": y1,
                "x_center": x_center,
                "y_center": y_center,
                "rectangle": Geometry.bbox_to_poly((x0, y0, x1, y1)),
            }
        )

    # Clustering.overlap 現在接受 bbox tuple，並回傳索引群組
    bboxes:List[Tuple[float,float,float,float]] = [
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
            GroupItem(
                box=items[g_sorted_idxs[0]]["rectangle"],
                contents=texts,
                rectangles={"counts": len(g_sorted_idxs), "items": rects},
            )
        )

    content = "\n".join(all_contents)
    return GroupedCorpus(content=content, group=result_groups)
