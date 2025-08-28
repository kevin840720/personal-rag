from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .types import PageImage
from .ops import ImageOps


def _enhance_target_image(
    img: Image.Image,
    target: str = "white",
    threshold: int = 200,
    boost: int = 30,
    other_scale: float = 0.7,
    grey_gap: int | None = None,
) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.int16)
    if target == "white":
        mask_thr = ImageOps._build_threshold_mask(arr, threshold, mode="high")
        mask_gray = ImageOps._near_gray_mask(arr, grey_gap)
        mask = mask_thr & mask_gray
        arr[mask] = np.clip(arr[mask] + boost, 0, 255)
    elif target == "black":
        mask_thr = ImageOps._build_threshold_mask(arr, threshold, mode="low")
        mask_gray = ImageOps._near_gray_mask(arr, grey_gap)
        mask = mask_thr & mask_gray
        arr[mask] = np.clip(arr[mask] - boost, 0, 255)
    else:
        raise ValueError("target 應為 'white' 或 'black'")
    arr[~mask] = np.clip(arr[~mask] * other_scale, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _invert_image(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.uint8)
    return Image.fromarray(255 - arr)


def _mask_and_set_rgb_image(
    img: Image.Image,
    threshold: int,
    mode: str = "high",
    set_rgb: tuple[int, int, int] = (255, 255, 255),
    check=("R", "G", "B"),
) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    mask = ImageOps._build_threshold_mask(arr, threshold=threshold, mode=mode, check=check)
    arr[mask] = set_rgb
    return Image.fromarray(arr)


def preprocess_white_branch(page:PageImage) -> Tuple[PageImage, PageImage]:
    # 01U: grey enhanced (two-pass)
    grey_img = _enhance_target_image(page.image, target="white", threshold=80, boost=0, other_scale=1.0, grey_gap=20)
    grey_img = _enhance_target_image(grey_img, target="white", threshold=200, boost=30, other_scale=0.7, grey_gap=None)
    grey_page = PageImage(grey_img, page.filename, page.page, variant="01U_grey")

    # 01L: white enhanced
    white_img = _enhance_target_image(page.image, target="white", threshold=200, boost=0, other_scale=1.0, grey_gap=None)
    white_page = PageImage(white_img, page.filename, page.page, variant="01L_white")

    # 02U / 02L: invert
    grey_inv_page = PageImage(_invert_image(grey_page.image), page.filename, page.page, variant="02U_invert")
    white_inv_page = PageImage(_invert_image(white_page.image), page.filename, page.page, variant="02L_invert")
    return grey_inv_page, white_inv_page


def preprocess_color_branch(page: PageImage) -> PageImage:
    # 11: color filter to BW
    bw_img = _enhance_target_image(page.image, target="white", threshold=0, boost=255, other_scale=1.0, grey_gap=20)
    bw_img = _mask_and_set_rgb_image(bw_img, threshold=254, mode="low", set_rgb=(0, 0, 0), check=("R", "G", "B"))
    return PageImage(bw_img, page.filename, page.page, variant="11_white2black")


def preprocess_white_textbook(page: PageImage) -> PageImage:
    # 51: color filter -> force high channels to white (single pass)
    img = _mask_and_set_rgb_image(page.image, threshold=200, mode="high", set_rgb=(255, 255, 255), check=("R", "G", "B"))
    return PageImage(img, page.filename, page.page, variant="51_color_filter")


def preprocess_color_notes(page: PageImage) -> PageImage:
    # Alias of color branch used in main2.py (61 pipeline)
    return preprocess_color_branch(page)
