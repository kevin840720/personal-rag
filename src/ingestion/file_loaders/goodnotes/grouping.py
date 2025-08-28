from __future__ import annotations

from typing import List, Dict

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
                "img": r.crop.page.page_id,
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

    groups = Clustering.cluster_by_overlap(items, height_ratio=2.0, width_ratio=None)

    def group_key(g):
        min_y = min(item["y_center"] for item in g)
        min_x = min(item["x_center"] for item in g if item["y_center"] == min_y)
        return (min_y, min_x)

    groups_sorted = sorted(groups, key=group_key)
    result_groups: List[GroupItem] = []
    all_contents: List[str] = []
    for g in groups_sorted:
        g_sorted = sorted(g, key=lambda d: (d["y_center"], d["x_center"]))
        rects = [it["rectangle"] for it in g_sorted]
        texts = [it["rec_text"] for it in g_sorted]
        all_contents.extend(texts)
        result_groups.append(
            GroupItem(
                box=g_sorted[0]["rectangle"],
                contents=texts,
                rectangles={"counts": len(g_sorted), "items": rects},
            )
        )

    content = "\n".join(all_contents)
    return GroupedCorpus(content=content, group=result_groups)

