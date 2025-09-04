"""
@File    :  test_grouping.py
@Time    :  2025/08/29 15:10:00
@Author  :  Kevin Wang
@Desc    :  Unit tests for GroupingOps
"""

from __future__ import annotations

from PIL import Image

from ingestion.file_loaders.goodnotes.grouping import GroupingOps
from ingestion.file_loaders.goodnotes.types import (PageImage,
                                                    CropItem,
                                                    RecItem,
                                                    RecognitionResult,
                                                    )


def make_page(w=60,
              h=40,
              color=(255, 255, 255),
              filename="demo.pdf",
              page=1,
              ) -> PageImage:
    img = Image.new("RGB", (w, h), color=color)
    return PageImage(img, filename, page)


class TestGroupingOps:
    def test_group_into_corpus(self):
        """測試 group_into_corpus 的分群與內容彙整。

        - 構造三個裁切：A 與 B 互相重疊（同一列），C 遠離；
        - 預期 (A,B) 會被分為一組，C 自成一組；
        - `content` 以閱讀順序串接為 "A\nB\nC"。
        """
        page = make_page(60, 40)  # 不重要，隨便產生就好

        # Make 3 crops: A and B overlap (same row), C far away
        crop_a = CropItem(page=page,
                          box=(5, 5, 15, 12),
                          poly_origin=[[5, 5], [15, 12]],
                          det_score=0.9,
                          image=page.image.crop((5, 5, 15, 12)),
                          expand_ratio=1.0,
                          )
        crop_b = CropItem(page=page,
                          box=(14, 5, 24, 12),
                          poly_origin=[[14, 5], [24, 12]],
                          det_score=0.9,
                          image=page.image.crop((14, 5, 24, 12)),
                          expand_ratio=1.0,
                          )
        crop_c = CropItem(page=page,
                          box=(40, 30, 50, 38),
                          poly_origin=[[40, 30], [50, 38]],
                          det_score=0.9,
                          image=page.image.crop((40, 30, 50, 38)),
                          expand_ratio=1.0,
                          )

        rec = RecognitionResult(items=[RecItem(crop=crop_a, text="A", score=0.9),
                                       RecItem(crop=crop_b, text="B", score=0.9),
                                       RecItem(crop=crop_c, text="C", score=0.9),
                                       ])

        corpus_cm = GroupingOps.group_into_corpus(rec)

        assert corpus_cm.content.count("\n") == 2
        assert corpus_cm.content.split("\n") == ["A", "B", "C"]
        assert len(corpus_cm.group) == 2  # (A,B) grouped; C alone

        # 內容與群數符合預期
        assert corpus_cm.content == "A\nB\nC"
