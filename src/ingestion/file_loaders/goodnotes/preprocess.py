from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ingestion.file_loaders.goodnotes.types import PageImage
from ingestion.file_loaders.goodnotes.ops import ImageOps


class GoodNotesPreprocessOps:
    """GoodNotes 的高階前處理流程

    - 只負責「流程編排」，不重寫影像計算；
    - 低階運算全部交給 `ImageOps`（如增強、反白、遮罩設色）；
    """

    @classmethod
    def preprocess_black_notes_text(cls, page:PageImage) -> Tuple[PageImage,PageImage]:
        """白底強化分支，回傳兩張「反白」後的影像。

        做法：
        1) 針對接近灰白的區域做兩次增強，得到較乾淨的灰增強版本（01U）。
        2) 針對白色做單次增強，得到白增強版本（01L）。
        3) 將兩個版本各自反白，輸出為 02U/02L 兩個變體。

        Args:
            page (PageImage): 輸入頁面（含 PIL 圖與檔名、頁碼標識）。

        Returns:
            Tuple[PageImage, PageImage]:
                - 第一張：灰增強後再反白（variant="02U_invert"）。
                - 第二張：白增強後再反白（variant="02L_invert"）。
        """
        # 01U: grey enhanced (two-pass)
        base_arr = np.array(page.image.convert("RGB"))
        # 先將灰色與白色以外的色調修正成黑色
        pass1 = ImageOps.enhance_target(base_arr,
                                        target="white",
                                        threshold=80,
                                        boost=0,
                                        other_scale=0.0,
                                        grey_gap=20,
                                        )
        # 再將灰色修正的更深
        pass2 = ImageOps.enhance_target(pass1,
                                        target="white",
                                        threshold=200,
                                        boost=30,
                                        other_scale=0.3,
                                        grey_gap=None,
                                        )
        grey_img = Image.fromarray(pass2)
        grey_page = PageImage(grey_img,
                              page.filename,
                              page.page,
                              variant="01U_grey",
                              )

        # 01L: white enhanced
        white_pass = ImageOps.enhance_target(base_arr,
                                             target="white",
                                             threshold=200,
                                             boost=0,
                                             other_scale=0.0,
                                             grey_gap=None,
                                             )
        white_img = Image.fromarray(white_pass)
        white_page = PageImage(white_img,
                              page.filename,
                              page.page,
                              variant="01L_white",
                              )

        # 02U / 02L: invert
        grey_inv_arr = ImageOps.invert_image(np.array(grey_page.image.convert("RGB")))
        grey_inv_page = PageImage(Image.fromarray(grey_inv_arr),
                                  page.filename,
                                  page.page,
                                  variant="02U_invert",
                                  )
        white_inv_arr = ImageOps.invert_image(np.array(white_page.image.convert("RGB")))
        white_inv_page = PageImage(Image.fromarray(white_inv_arr),
                                  page.filename,
                                  page.page,
                                  variant="02L_invert",
                                  )
        return grey_inv_page, white_inv_page

    @classmethod
    def preprocess_black_notes_handwriting(cls, page:PageImage) -> PageImage:
        """彩色筆記轉成黑白風格。

        流程：
        1) 強化灰階的像素全數轉換成白色；
        2) 將接近純白以外的像素統一壓到黑色，形成接近二值化的效果。

        Args:
            page (PageImage): 輸入頁面（含 PIL 圖與檔名、頁碼標識）。

        Returns:
            PageImage: 處理後的頁面，`variant="11_color2black"`。
        """
        # 11: color filter to BW
        base_arr = np.array(page.image.convert("RGB"))
        # 灰階轉成白色
        enhanced = ImageOps.enhance_target(base_arr,
                                           target="white",
                                           threshold=0,
                                           boost=255,
                                           other_scale=1.0,
                                           grey_gap=20,
                                           )
        # 非白轉黑色
        bw_arr = ImageOps.mask_and_set_rgb(enhanced,
                                           threshold=254,
                                           mode="low",
                                           set_rgb=(0, 0, 0),
                                           check=("R"),
                                           )
        bw_arr = ImageOps.mask_and_set_rgb(bw_arr,
                                           threshold=254,
                                           mode="low",
                                           set_rgb=(0, 0, 0),
                                           check=("G"),
                                           )
        bw_arr = ImageOps.mask_and_set_rgb(bw_arr,
                                           threshold=254,
                                           mode="low",
                                           set_rgb=(0, 0, 0),
                                           check=("B"),
                                           )
        bw_img = Image.fromarray(bw_arr)
        return PageImage(bw_img,
                         page.filename,
                         page.page,
                         variant="11_color2black",
                         )

    @classmethod
    def preprocess_white_textbook_text(cls, page:PageImage) -> PageImage:
        """適合白底教科書：把彩色筆記拉到純白。

        將 RGB 值高於門檻的像素直接設為 (255,255,255)，清乾淨背景、保留深色文字。

        Args:
            page (PageImage): 輸入頁面（含 PIL 圖與檔名、頁碼標識）。

        Returns:
            PageImage: 處理後的頁面，`variant="51_color_filter"`。
        """
        # 51: color filter -> force high channels to white (single pass)
        base_arr = np.array(page.image.convert("RGB"))
        bw_arr = ImageOps.mask_and_set_rgb(base_arr,
                                           threshold=200,
                                           mode="high",
                                           set_rgb=(255,255,255),
                                           check=("R"),
                                           )
        bw_arr = ImageOps.mask_and_set_rgb(bw_arr,
                                           threshold=200,
                                           mode="high",
                                           set_rgb=(255,255,255),
                                           check=("G"),
                                           )
        bw_arr = ImageOps.mask_and_set_rgb(bw_arr,
                                           threshold=200,
                                           mode="high",
                                           set_rgb=(255,255,255),
                                           check=("B"),
                                           )
        img = Image.fromarray(bw_arr)
        return PageImage(img,
                         page.filename,
                         page.page,
                         variant="51_color_filter",
                         )

    @classmethod
    def preprocess_white_textbook_handwriting(cls, page:PageImage) -> PageImage:
        # 61: 只保留非灰色階，並轉成黑白頁面，同「黑底筆記處理」的原則
        return cls.preprocess_black_notes_handwriting(page)
