# -*- encoding: utf-8 -*-
"""
@File    :  pdf.py
@Time    :  2025/01/21 16:06:08
@Author  :  Kevin Wang
@Desc    :  針對 Goodnotes 輸出的 pdf/image 的 Loader
"""

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageStat
from docling_core.types import DoclingDocument

from ingestion.base import DocumentLoader, LoaderResult
from ingestion.utils import export_to_user_text
from ingestion.file_loaders.image import DoclingImageLoader
from ingestion.file_loaders.pdf import DoclingPDFLoader
from objects import PDFMetadata


class GoodnotesMetadata(PDFMetadata):  # TODO: 考慮要不要放到 objects.py
    page:int
    outlines:List[str]

    def to_dict(self) -> Dict[str, Union[str,int,float,List,Dict]]:
        result = super().to_dict()
        result['page'] = self.page
        result['outlines'] = self.outlines
        return result

    @classmethod
    def from_dict(cls,
                  data:Dict[str, Union[str,int,float,List,Dict]],
                  ) -> "GoodnotesMetadata":
        base = PDFMetadata.from_dict(data)
        return cls(file_type=base.file_type,
                   file_name=base.file_name,
                   title=base.title,
                   author=base.author,
                   subject=base.subject,
                   created_at=base.created_at,
                   modified_at=base.modified_at,
                   source=base.source,
                   producer=base.producer,
                   page=data['page'],
                   outlines=data['outlines'],
                   )
    
class DoclingGoodnotesLoader(DocumentLoader):
    def __init__(self):
        super().__init__()
        self._image_loader = DoclingImageLoader(do_ocr=True)
        self._pdf_loader = DoclingPDFLoader(do_ocr=False, do_table_structure=False)

    def _get_metadata(self,
                      path:Union[Path, str],
                      ) -> PDFMetadata:
        """這只會輸出對 pdf 的 metadata，不會輸出 outline

        Args:
            path (Union[Path, str]): _description_

        Returns:
            PDFMetadata: _description_
        """
        pdf_metadata:PDFMetadata = self._pdf_loader._get_metadata(path)
        return pdf_metadata

    def _get_metadata_w_outline(self,
                                path:Union[Path, str],
                                ) -> Dict[int,GoodnotesMetadata]:
        reader = PdfReader(path)
        base_meta_dict:dict = self._get_metadata(path).to_dict()

        page_outline_map:Dict[int,List[str]] = defaultdict(list)
        for outline in reader.outline:  # NOTE: 目前觀察到 Goodnotes 都只有單層 outline
            title = outline.title
            page_num = reader.get_destination_page_number(outline) + 1  # 從0開始
            page_outline_map[page_num].append(title)

        metadatas:Dict[int,GoodnotesMetadata] = {}
        for page_num in range(1, len(reader.pages)+1):
            outlines = page_outline_map[page_num]
            metadatas[page_num] = GoodnotesMetadata(**base_meta_dict,
                                                    page=page_num,
                                                    outlines=outlines,
                                                    )
        return metadatas

    def _pdf_to_page_images(self,
                            pdf_path:Union[str,Path],
                            outdir:Union[str,Path],
                            dpi:int=400,
                            fmt:str="png",
                            ) -> List[Path]:
        """
        將 PDF 轉成逐頁圖片
        回傳各頁輸出檔路徑（已排序）
        """
        pdf_path = Path(pdf_path)
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        pages = convert_from_path(str(pdf_path), dpi=dpi)
        page_paths: List[Path] = []
        for i, page in enumerate(pages, start=1):
            p = outdir / f"page-{i}.{fmt}"
            page.save(str(p))
            page_paths.append(p)
        return page_paths

    def _split_page_to_images(self,
                              page_img_path:Union[str,Path],
                              outdir:Union[str,Path],
                              grid_x:int=4,
                              grid_y:int=5,
                              vertical_overlap_ratio:float=0.10,
                              horizontal_overlap_ratio:float=0.50,
                              pad_x_ratio:float=0.10,
                              pad_y_ratio:float=0.02,
                              ) -> List[Path]:
        """
        將單一頁圖片加邊框後切成多張，含重疊
        回傳此頁所有切圖路徑（已排序）
        """
        page_img_path = Path(page_img_path)
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

        page = Image.open(page_img_path)
        mean_brightness = ImageStat.Stat(page.convert("L")).mean[0]
        fill_color = (255, 255, 255) if mean_brightness >= 128 else (0, 0, 0)

        ori_w, ori_h = page.size
        pad_x, pad_y = int(ori_w * pad_x_ratio), int(ori_h * pad_y_ratio)
        padded = ImageOps.expand(page, border=(pad_x, pad_y, pad_x, pad_y), fill=fill_color)

        w, h = padded.size
        cell_w, cell_h = w // grid_x, h // grid_y

        # 從檔名抓頁碼
        stem = page_img_path.stem  # e.g., "page-3"
        try:
            page_idx = int(stem.split("-")[-1])
        except Exception:
            page_idx = 1

        tiles: List[Path] = []
        for row in range(grid_y):
            for col in range(grid_x):
                left = max(int(col * cell_w - cell_w * horizontal_overlap_ratio), 0)
                right = min(int((col + 1) * cell_w + cell_w * horizontal_overlap_ratio), w)
                upper = max(int(row * cell_h - cell_h * vertical_overlap_ratio), 0)
                lower = min(int((row + 1) * cell_h + cell_h * vertical_overlap_ratio), h)
                crop = padded.crop((left, upper, right, lower))

                outpath = outdir / f"p{page_idx}-r{row}-c{col}.png"
                crop.save(str(outpath))
                tiles.append(outpath)
        return tiles

    def load(self, path:Union[str,Path]) -> List[LoaderResult]:

        path = Path(path)

        doclingdoc_dict, content_dict = dict(), dict()
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # 1) PDF→頁圖
            page_imgs = self._pdf_to_page_images(path, tmpdir / "pages", dpi=400)

            # 2) 每頁→多張切圖，逐張送入 OCR
            tiles_dir = tmpdir / "tiles"
            for page_idx, pp in enumerate(page_imgs):  # pp for page_path
                tile_paths = self._split_page_to_images(pp,
                                                        tiles_dir,
                                                        grid_x=4,
                                                        grid_y=5,
                                                        vertical_overlap_ratio=0.10,
                                                        horizontal_overlap_ratio=0.50,
                                                        pad_x_ratio=0.10,
                                                        pad_y_ratio=0.02,
                                                        )
                img_docs = []
                for tp in tile_paths:  # tp for tile_path
                    img_doc:DoclingDocument = self._image_loader.converter.convert(str(tp)).document
                    img_docs.append(img_doc)

                # 合併同頁 DoclingDocument
                merged_doc = DoclingDocument.concatenate(img_docs)
                content = export_to_user_text(merged_doc)

                doclingdoc_dict[page_idx+1] = merged_doc
                content_dict[page_idx+1] = content

        metas:Dict[int,GoodnotesMetadata] = self._get_metadata_w_outline(path)

        results:List[LoaderResult] = []
        for page_num in metas:
            results.append(
                LoaderResult(content=content_dict[page_num],
                             metadata=metas[page_num],
                             doc=doclingdoc_dict[page_num],
                             )
            )

        return results

