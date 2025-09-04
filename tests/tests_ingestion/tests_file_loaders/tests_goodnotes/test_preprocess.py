"""
@File    :  test_preprocess.py
@Time    :  2025/08/29 10:30:00
@Author  :  Kevin Wang
@Desc    :  Unit tests for GoodNotes preprocess helpers
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from ingestion.file_loaders.goodnotes.types import PageImage
from ingestion.file_loaders.goodnotes.preprocess import GoodNotesPreprocessOps


def make_page(arr:np.ndarray,
              filename:str="demo.pdf",
              page:int=1,
              ) -> PageImage:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return PageImage(img, filename, page)


class TestGoodnotesPreprocessOps:
    def test_preprocess_black_notes_text(self):
        arr = np.array([[[210, 210, 210], [200, 255, 255]],
                        [[220, 220, 220], [ 50,  60,  70]],
                        ], dtype=np.uint8)
        page = make_page(arr, filename="demo.pdf", page=1)
        grey_inv, white_inv = GoodNotesPreprocessOps.preprocess_black_notes_text(page)

        # Variants and identity preserved
        assert grey_inv.variant == "02U_invert"
        assert white_inv.variant == "02L_invert"
        assert grey_inv.filename == page.filename and grey_inv.page == page.page
        assert white_inv.filename == page.filename and white_inv.page == page.page

        # Images are PIL and same size as input
        assert isinstance(grey_inv.image, Image.Image)
        assert isinstance(white_inv.image, Image.Image)
        assert grey_inv.image.size == page.image.size
        assert white_inv.image.size == page.image.size


        # Value
        grey_expect = Image.fromarray(
            np.array([[[15, 15, 15], [255, 255, 255]],
                      [[ 5,  5,  5], [255, 255, 255]]],
                     dtype=np.uint8,
                     )
            )
        assert np.array_equal(np.array(grey_inv.image), np.array(grey_expect))
        white_expect = Image.fromarray(
            np.array([[[45, 45, 45], [ 55,   0,   0]],
                      [[35, 35, 35], [255, 255, 255]]],
                     dtype=np.uint8,
                     )
            )
        assert np.array_equal(np.array(white_inv.image), np.array(white_expect))

    def test_preprocess_black_notes_handwriting(self):
        # Mixed colors; expect a black/white result (values in {0,255})
        arr = np.array([[[ 10,   0,  20], [255,   0,   0]],
                        [[120,  90,  80], [250, 250, 250]],
                        ],
                       dtype=np.uint8,
                       )
        page = make_page(arr, filename="demo.pdf", page=2)

        out = GoodNotesPreprocessOps.preprocess_black_notes_handwriting(page)
        assert out.variant == "11_color2black"
        assert out.filename == page.filename and out.page == page.page

        expect = Image.fromarray(
            np.array([[[255, 255, 255], [  0,   0,   0]],
                      [[  0,   0,   0], [255, 255, 255]]],
                     dtype=np.uint8,
                     )
            )
        assert np.array_equal(np.array(out.image), np.array(expect))

    def test_preprocess_white_textbook_variant(self):
        arr = np.array([[[ 10,   0,  20], [255,   0,   0]],
                        [[120,  90,  80], [250, 250, 250]],
                        ],
                       dtype=np.uint8,
                       )
        page = make_page(arr, filename="demo.pdf", page=3)

        out = GoodNotesPreprocessOps.preprocess_white_textbook_text(page)
        assert out.variant == "51_color_filter"
        assert out.filename == page.filename and out.page == page.page

        expect = Image.fromarray(
            np.array([[[ 10,   0,  20], [255, 255, 255]],
                      [[120,  90,  80], [255, 255, 255]]],
                     dtype=np.uint8,
                     )
            )
        assert np.array_equal(np.array(out.image), np.array(expect))

    def test_preprocess_white_textbook_handwriting(self):
        # Mixed colors; expect a black/white result (values in {0,255})
        arr = np.array([[[ 10,   0,  20], [255,   0,   0]],
                        [[120,  90,  80], [250, 250, 250]],
                        ],
                       dtype=np.uint8,
                       )
        page = make_page(arr, filename="demo.pdf", page=2)

        out = GoodNotesPreprocessOps.preprocess_black_notes_handwriting(page)
        assert out.variant == "11_color2black"
        assert out.filename == page.filename and out.page == page.page

        expect = Image.fromarray(
            np.array([[[255, 255, 255], [  0,   0,   0]],
                      [[  0,   0,   0], [255, 255, 255]]],
                     dtype=np.uint8,
                     )
            )
        assert np.array_equal(np.array(out.image), np.array(expect))