"""
@File    :  test_ops.py
@Time    :  2025/08/29 06:55:00
@Author  :  Kevin Wang
@Desc    :
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import numpy as np

from ingestion.file_loaders.goodnotes.ops import ImageOps, Geometry


def make_image_file(arr:np.ndarray,
                    out_path:str="in.png",
                    ) -> Path:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    img.save(out_path)

class TestImageOps:
    def test_ensure_channels(self):
        assert ImageOps._ensure_channels(["R"]) == [0]
        assert ImageOps._ensure_channels(["R", "B"]) == [0, 2]
        # None or empty → default to RGB
        assert ImageOps._ensure_channels(None) == [0, 1, 2]
        assert ImageOps._ensure_channels(()) == [0, 1, 2]

    def test_build_threshold_mask_high(self):
        # 2x2 pixels
        arr = np.array([[[200, 200, 200], [50, 50, 50]],
                        [[150, 150, 150], [250, 250, 250]],
                        ],
                       dtype=np.uint8,
                       )

        # High: all checked channels > threshold
        mask_high = ImageOps._build_threshold_mask(arr,
                                                   threshold=180,
                                                   mode="high",
                                                   check=("R", "G", "B"),
                                                   )

        # Expect True at (0,0) and (1,1)
        assert mask_high.tolist() == [[True, False], [False, True]]

    def test_build_threshold_mask_low(self):
        # 2x2 pixels
        arr = np.array([[[200, 200, 200], [50, 50, 50]],
                        [[150, 150, 150], [250, 250, 250]],
                        ],
                       dtype=np.uint8,
                       )

        # Low: all checked channels < threshold
        mask_low = ImageOps._build_threshold_mask(arr,
                                                  threshold=100,
                                                  mode="low",
                                                  check=("R", "G", "B"),
                                                  )
    
        # Expect True only at (0,1)
        assert mask_low.tolist() == [[False, True], [False, False]]

        # Check specific channel only (e.g., R)
        arr2 = np.array([[[120, 50, 50],
                          [90, 200, 200]],
                         ],
                        dtype=np.uint8,
                        )
        mask_r_high = ImageOps._build_threshold_mask(arr2,
                                                     threshold=100,
                                                     mode="high",
                                                     check=("R",),
                                                     )
        assert mask_r_high.tolist() == [[True, False]]

    def test_near_gray_mask(self):
        arr = np.array([[[100, 100, 100], [100, 110, 105]],   # grey diffs: 0, 10
                        [[100, 120, 140], [10, 240, 50]],     # grey diffs: 40, 230
                        ],
                       dtype=np.uint8,
                       )

        # grey_gap=None → all True
        mask_all = ImageOps._near_gray_mask(arr, None)
        assert mask_all.all()

        mask10 = ImageOps._near_gray_mask(arr, 10)
        assert mask10.tolist() == [[True, True],
                                   [False, False],
                                   ]

    def test_enhance_target_white(self):
        # 2x2 image: two pixels above threshold (near grey), others below or not grey
        arr = np.array([[[210, 210, 210], [200, 255, 255]],
                        [[220, 220, 220], [ 50,  60,  70]],
                        ],
                       dtype=np.uint8,
                       )
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            in_path = tmpdir / "in_white.png"
            out_path = tmpdir / "out_white.png"

            make_image_file(arr, in_path)  # 製作測試資料

            ImageOps.enhance_target(in_path,
                                    out_path,
                                    target="white",
                                    threshold=200,
                                    boost=30,
                                    other_scale=0.5,
                                    grey_gap=10,
                                    )

            got = np.array(Image.open(out_path).convert("RGB"))

            # Manually compute expected using same logic
            expect = np.array([[[240, 240, 240], [100, 127, 127]],
                               [[250, 250, 250], [ 25,  30,  35]],
                               ],
                              dtype=np.uint8,
                              )
        assert np.array_equal(got, expect)

    def test_enhance_target_black(self):
        arr = np.array([[[30, 30, 30], [100, 100, 100]],
                        [[20, 20, 20], [ 80,  70,  60]],
                        ],
                       dtype=np.uint8,
                       )
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            in_path = tmpdir / "in_black.png"
            out_path = tmpdir / "out_black.png"

            make_image_file(arr, in_path)  # 製作測試資料

            ImageOps.enhance_target(in_path,
                                    out_path,
                                    target="black",
                                    threshold=40,
                                    boost=10,
                                    other_scale=2.0,
                                    grey_gap=10,
                                    )

            got = np.array(Image.open(out_path).convert("RGB"))

        # Manually compute expected using same logic
        expect = np.array([[[20, 20, 20], [200, 200, 200]],
                           [[10, 10, 10], [160, 140, 120]],
                           ],
                          dtype=np.uint8,
                          )
        assert np.array_equal(got, expect)

    def test_invert_image(self):
        arr = np.array([[[ 0, 128, 255],
                         [10,  20,  30]
                         ]],
                       dtype=np.uint8,
                       )
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            in_path = tmpdir / "in_inv.png"
            out_path = tmpdir / "out_inv.png"

            make_image_file(arr, in_path)

            ImageOps.invert_image(in_path, out_path)
            got = np.array(Image.open(out_path).convert("RGB"))

        expect = np.array([[[255, 127,   0],
                            [245, 235, 225]
                            ]],
                           dtype=np.uint8,
                           )
        assert np.array_equal(got, expect)

    def test_mask_and_set_rgb(self):
        arr = np.array([[[ 50, 50, 50], [200, 200, 200]],
                        [[150, 80, 80], [ 80, 150,  80]],
                        ],
                       dtype=np.uint8,
                       )
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            in_path = tmpdir / "in_mask.png"
            out_path = tmpdir / "out_mask.png"

            make_image_file(arr, in_path)

            # Set red where R channel > 100, regardless of G/B
            ImageOps.mask_and_set_rgb(in_path,
                                      out_path,
                                      threshold=100,
                                      mode="high",
                                      set_rgb=(255, 0, 0),
                                      check=("R",),
                                      )

            got = np.array(Image.open(out_path).convert("RGB"))

        # Expected: (0,0) unchanged; (0,1) set to red; (1,0) set to red; (1,1) unchanged
        expect = np.array([[[ 50, 50, 50], [255,   0,   0]],
                           [[255,  0,  0], [ 80, 150,  80]],
                           ],
                          dtype=np.uint8,
                          )
        assert np.array_equal(got, expect)

    def test_estimate_background_color(self):
        # White-ish image
        white = Image.fromarray(np.full((2, 2, 3), 220, dtype=np.uint8), mode="RGB")
        assert ImageOps.estimate_background_color(white, threshold=128) == "white"

        # Black-ish image
        black = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB")
        assert ImageOps.estimate_background_color(black, threshold=128) == "black"

        # Boundary equals threshold → white
        mid = Image.fromarray(np.full((1, 1, 3), 128, dtype=np.uint8), mode="RGB")
        assert ImageOps.estimate_background_color(mid, threshold=128) == "white"
