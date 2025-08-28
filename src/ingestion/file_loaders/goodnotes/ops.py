"""GoodNotes 影像/幾何/分群的小工具集合。

偏實用導向，描述都用白話文，方便後續維護的人快速上手。
"""

from __future__ import annotations

from pathlib import Path
from typing import (Dict,
                    Iterable,
                    Iterator,
                    List,
                    Optional,
                    Tuple,
                    Union,
                    )
import gc

from PIL import Image
import numpy as np

from ingestion.file_loaders.goodnotes.types import PageImage

class ImageOps:
    """影像處理的工具（針對 RGB 圖）"""
    CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}

    @classmethod
    def _ensure_channels(cls,
                         check:Optional[Iterable[str]],
                         ) -> List[int]:
        """把通道名稱（R/G/B）轉成索引。

        Args:
            check (Optional[Iterable[str]]): 要檢查的通道名稱，例如 ("R", "G", "B")。
                                             傳入 None 或空值時，視為全 RGB。

        Returns:
            List[int]: 對應到通道的索引列表，例如 [0, 1, 2]。
        """
        if not check:
            check = ("R", "G", "B")
        return [cls.CHANNEL_INDEX[c] for c in check]

    @classmethod
    def _build_threshold_mask(cls,
                              arr:np.ndarray,
                              threshold:int,
                              mode:str="high",
                              check:Optional[Iterable[str]]=("R", "G", "B"),
                              ) -> np.ndarray:
        """依門檻建立布林遮罩，挑出「很亮」或「很暗」的像素。

        Args:
            arr (np.ndarray): 影像陣列，形狀 H×W×3，RGB。
            threshold (int): 門檻值（0–255）。
            mode (str, optional): 'high' 表示所有指定通道都要大於門檻；'low' 表示都要小於門檻。預設為 'high'。
            check (Optional[Iterable[str]], optional): 要檢查的通道，預設為 ("R", "G", "B")。

        Raises:
            ValueError: 當 `mode` 不是 'high' 或 'low' 時。

        Returns:
            np.ndarray: 形狀 H×W 的布林遮罩。
        """
        idxs = cls._ensure_channels(check)
        if mode == "high":
            return (arr[..., idxs] > threshold).all(axis=2)
        if mode == "low":
            return (arr[..., idxs] < threshold).all(axis=2)
        raise ValueError("mode 應為 'high' 或 'low'")

    @classmethod
    def _near_gray_mask(cls,
                        arr:np.ndarray,
                        grey_gap:Optional[int],
                        ) -> np.ndarray:
        """找出接近灰階的像素。

        Args:
            arr (np.ndarray): 影像陣列，形狀 H×W×3，RGB。
            grey_gap (Optional[int]): 允許的 RGB 最大最小差；None 表示不限制（全部視為灰階）。

        Returns:
            np.ndarray: 形狀 H×W 的布林遮罩，True 表示近灰色。
        """
        if grey_gap is None:
            return np.ones(arr.shape[:2], dtype=bool)
        diff = np.max(arr, axis=2) - np.min(arr, axis=2)
        return diff <= grey_gap

    @classmethod
    def enhance_target(cls,
                       img_path:Union[str,Path],
                       out_path:Union[str,Path],
                       target:str="white",
                       threshold:int=200,
                       boost:int=30,
                       other_scale:float=0.7,
                       grey_gap:Optional[int]=None,
                       ) -> None:
        """強化白/黑目標區域，其他像素做縮放。

        Args:
            img_path (Union[str, Path]): 輸入影像路徑。
            out_path (Union[str, Path]): 輸出影像路徑。
            target (str, optional): 'white' 強化亮灰像素、'black' 強化暗灰像素。預設 'white'。
            threshold (int, optional): 亮/暗的門檻值。預設 200。
            boost (int, optional): 目標像素的加/減量級。預設 30。
            other_scale (float, optional): 非目標像素的縮放係數。預設 0.7。
            grey_gap (Optional[int], optional): 灰階容許差。None 表示不限制。

        Raises:
            ValueError: 當 `target` 不是 'white' 或 'black' 時。

        Returns:
            None
        """
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.int16)
        if target not in ("white", "black"):
            raise ValueError("target 應為 'white' 或 'black'")
        if target == "white":
            mask_thr = cls._build_threshold_mask(arr, threshold, mode="high")
            mask_gray = cls._near_gray_mask(arr, grey_gap)
            mask = mask_thr & mask_gray
            arr[mask] = np.clip(arr[mask] + boost, 0, 255)
        else:
            mask_thr = cls._build_threshold_mask(arr, threshold, mode="low")
            mask_gray = cls._near_gray_mask(arr, grey_gap)
            mask = mask_thr & mask_gray
            arr[mask] = np.clip(arr[mask] - boost, 0, 255)
        arr[~mask] = np.clip(arr[~mask] * other_scale, 0, 255)
        Image.fromarray(arr.astype(np.uint8)).save(out_path)

    @classmethod
    def invert_image(cls,
                     img_path:Union[str,Path],
                     out_path:Union[str,Path],
                     ) -> None:
        """影像反白（把每個通道做 255 減原值）。

        Args:
            img_path (Union[str, Path]): 輸入影像路徑。
            out_path (Union[str, Path]): 輸出影像路徑。

        Returns:
            None
        """
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.uint8)
        arr = 255 - arr
        Image.fromarray(arr).save(out_path)

    @classmethod
    def mask_and_set_rgb(cls,
                         img_path:Union[str,Path],
                         out_path:Union[str,Path],
                         threshold:int,
                         mode:str="high",
                         set_rgb:Tuple[int,int,int]=(255,255,255),
                         check:Optional[Iterable[str]]=("R","G","B"),
                         ) -> None:
        """依門檻製作遮罩，將通過的像素設為指定顏色。

        Args:
            img_path (Union[str, Path]): 輸入影像路徑。
            out_path (Union[str, Path]): 輸出影像路徑。
            threshold (int): 門檻值（0–255）。
            mode (str, optional): 'high' 或 'low'，依據門檻比較方向。預設 'high'。
            set_rgb (Tuple[int, int, int], optional): 通過遮罩後要設的 RGB 顏色。
                預設 (255, 255, 255)。
            check (Optional[Iterable[str]], optional): 要檢查的通道。預設 RGB。

        Returns:
            None
        """
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        mask = cls._build_threshold_mask(arr, threshold=threshold, mode=mode, check=check)
        arr[mask] = set_rgb
        Image.fromarray(arr).save(out_path)

    @classmethod
    def estimate_background_color(cls,
                                  img:Image.Image,
                                  threshold:int=128,
                                  ) -> str:
        """以平均亮度粗估背景色。

        Args:
            img (PIL.Image.Image): PIL 影像物件。
            threshold (int, optional): 亮度門檻，平均亮度大於等於此值視為白底。
                預設 128。

        Returns:
            str: 'white' 或 'black'。
        """
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        channel_means = arr.reshape(-1, 3).mean(axis=0)
        brightness = float(channel_means.mean())
        return "white" if brightness >= threshold else "black"


class Geometry:
    """跟座標/框框（bbox/poly）相關的計算。"""
    @classmethod
    def poly_to_bbox(cls,
                     poly:List[List[int]],
                     ) -> Tuple[int,int,int,int]:
        """把四點多邊形轉成 (x_min, y_min, x_max, y_max)。

        Args:
            poly (List[List[int]]): 多邊形點列，格式 [[x, y], ...]。

        Returns:
            Tuple[int, int, int, int]: 對應的 bbox (x_min, y_min, x_max, y_max)。
        """
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def bbox_to_poly(cls,
                     bbox:Tuple[int,int,int,int],
                     ) -> List[List[int]]:
        """把 bbox 還原成四個角點（左下→左上→右上→右下）。

        Args:
            bbox (Tuple[int, int, int, int]): (x_min, y_min, x_max, y_max)。

        Returns:
            List[List[int]]: [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]]。
        """
        x_min, y_min, x_max, y_max = bbox
        return [[x_min, y_max],
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max]]

    @classmethod
    def _expand_range(cls,
                      min_v:float,
                      max_v:float,
                      length:Optional[float]=None,
                      ratio:Optional[float]=None,
                      pixel:Optional[float]=None,
                      ) -> Tuple[int,int]:
        """依比例/像素把一段 [min, max] 往外擴。

        Args:
            min_v (float): 原始區間最小值。
            max_v (float): 原始區間最大值。
            length (Optional[float]): 區間長度；None 則用 max_v - min_v。
            ratio (Optional[float]): 目標長度 = length * ratio。
            pixel (Optional[float]): 目標長度 = length + 2*pixel。

        Returns:
            Tuple[int, int]: 擴張後的新 (min, max)，取整數。
        """
        if length is None:
            length = max_v - min_v
        if ratio is not None:
            new_len = length * ratio
        elif pixel is not None:
            new_len = length + 2 * pixel
        else:
            new_len = length
        delta = int(round((new_len - length) / 2))
        return int(min_v) - delta, int(max_v) + delta

    @classmethod
    def expand_bbox(cls,
                    bbox:Tuple[int,int,int,int],
                    height_ratio:Optional[float]=None,
                    height_pixel:Optional[float]=None,
                    width_ratio:Optional[float]=None,
                    width_pixel:Optional[float]=None,
                    clip_size:Optional[Tuple[int,int]]=None,
                    ) -> Tuple[int,int,int,int]:
        """把 bbox 依比例/像素往外擴，必要時裁到圖片邊界內。

        Args:
            bbox (Tuple[int, int, int, int]): (x_min, y_min, x_max, y_max)。
            height_ratio (Optional[float]): 高度放大比例，與 `height_pixel` 擇一。
            height_pixel (Optional[float]): 高度兩側擴的像素量，與 `height_ratio` 擇一。
            width_ratio (Optional[float]): 寬度放大比例，與 `width_pixel` 擇一。
            width_pixel (Optional[float]): 寬度兩側擴的像素量，與 `width_ratio` 擇一。
            clip_size (Optional[Tuple[int, int]]): (W, H) 用來裁邊界；None 則不裁。

        Returns:
            Tuple[int, int, int, int]: 擴張後的 (x_min, y_min, x_max, y_max)。
        """
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        ex_xmin, ex_xmax = cls._expand_range(x_min, x_max, w, width_ratio, width_pixel)
        ex_ymin, ex_ymax = cls._expand_range(y_min, y_max, h, height_ratio, height_pixel)
        if clip_size is not None:
            W, H = clip_size
            ex_xmin = max(0, ex_xmin)
            ex_ymin = max(0, ex_ymin)
            ex_xmax = min(W, ex_xmax)
            ex_ymax = min(H, ex_ymax)
        return ex_xmin, ex_ymin, ex_xmax, ex_ymax

    @classmethod
    def is_bbox_overlap(cls,
                        a:Dict,
                        b:Dict,
                        height_ratio:Optional[float]=None,
                        height_pixel:Optional[float]=None,
                        width_ratio:Optional[float]=None,
                        width_pixel:Optional[float]=None,
                        ) -> bool:
        """判斷兩個 bbox 是否重疊（可選擇先擴張）。

        Args:
            a (Dict): 具有 x_min, y_min, x_max, y_max 的物件。
            b (Dict): 具有 x_min, y_min, x_max, y_max 的物件。
            height_ratio (Optional[float]): 高度擴張比例。
            height_pixel (Optional[float]): 高度擴張像素量。
            width_ratio (Optional[float]): 寬度擴張比例。
            width_pixel (Optional[float]): 寬度擴張像素量。

        Returns:
            bool: True 表示重疊，False 表示不重疊。
        """
        a_bbox = (a["x_min"], a["y_min"], a["x_max"], a["y_max"])
        b_bbox = (b["x_min"], b["y_min"], b["x_max"], b["y_max"])
        ax0, ay0, ax1, ay1 = cls.expand_bbox(a_bbox, height_ratio, height_pixel, width_ratio, width_pixel)
        bx0, by0, bx1, by1 = cls.expand_bbox(b_bbox, height_ratio, height_pixel, width_ratio, width_pixel)
        x_overlap = (ax0 <= bx1) and (bx0 <= ax1)
        y_overlap = (ay0 <= by1) and (by0 <= ay1)
        return x_overlap and y_overlap


class Clustering:
    """簡單的群組化工具（類 DBSCAN 與變形）。"""
    @classmethod
    def cluster_by_dbscan(
        cls,
        items: List[Dict],
        y_eps: int = 40,
        x_eps: int = 20,
        min_samples: int = 1,
    ) -> List[List[Dict]]:
        """用矩形距離做個很簡化的 DBSCAN 效果。

        - 以 `x_eps/y_eps` 當鄰近門檻，湊成小群組。
        - 盡量把靠近的點歸在一起，最後再依 y、x 平均排序。
        """
        if not items:
            return []
        coords = np.array([[d["x_center"], d["y_center"]] for d in items])
        n = len(items)
        visited = np.zeros(n, dtype=bool)
        clusters: List[List[Dict]] = []
        for i in range(n):
            if visited[i]:
                continue
            mask_x = np.abs(coords[:, 0] - coords[i, 0]) <= x_eps
            mask_y = np.abs(coords[:, 1] - coords[i, 1]) <= y_eps
            mask = mask_x & mask_y
            idxs = np.where(mask)[0]
            if len(idxs) >= min_samples:
                cluster: List[Dict] = []
                for idx in idxs:
                    if not visited[idx]:
                        visited[idx] = True
                        cluster.append(items[idx])
                clusters.append(cluster)
        for i in range(n):
            if not visited[i]:
                clusters.append([items[i]])
                visited[i] = True
        clusters.sort(
            key=lambda group: (
                float(np.mean([d["y_center"] for d in group])),
                float(np.mean([d["x_center"] for d in group])),
            )
        )
        return clusters

    @classmethod
    def cluster_by_ycenter(cls, items: List[Dict], y_thresh: int = 30) -> List[List[Dict]]:
        """只看 Y 向距離來分行，X 幾乎不限制。"""
        big = 10 ** 9
        return cls.cluster_by_dbscan(items, y_eps=y_thresh, x_eps=big, min_samples=1)

    @classmethod
    def cluster_by_overlap(
        cls,
        items: List[Dict],
        height_ratio: Optional[float] = None,
        height_pixel: Optional[float] = None,
        width_ratio: Optional[float] = None,
        width_pixel: Optional[float] = None,
    ) -> List[List[Dict]]:
        """以 bbox 是否互相重疊來群組（可帶擴張參數）。"""
        n = len(items)
        if n == 0:
            return []

        parents = list(range(n))

        def find(i: int) -> int:
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def union(i: int, j: int) -> None:
            pi, pj = find(i), find(j)
            if pi != pj:
                parents[pi] = pj

        for i in range(n):
            for j in range(i + 1, n):
                if Geometry.is_bbox_overlap(
                    items[i],
                    items[j],
                    height_ratio=height_ratio,
                    height_pixel=height_pixel,
                    width_ratio=width_ratio,
                    width_pixel=width_pixel,
                ):
                    union(i, j)

        groups_map: Dict[int, List[Dict]] = {}
        for idx in range(n):
            p = find(idx)
            groups_map.setdefault(p, []).append(items[idx])

        return list(groups_map.values())


class PdfOps:
    """和 PDF 轉圖有關的工具。"""
    @classmethod
    def pdf_page_images(
        cls,
        pdf_path: Union[str, Path],
        dpi: int = 600,
        fmt: str = "PNG",
        thread_count: int = 1,
    ) -> Iterator[PageImage]:
        """Converts a PDF to per-page images and yields them one by one.

        - `dpi` defaults to 600 (semantic), actual conversion uses `run_dpi` (default 1000).
        - After each yield, closes intermediate images and hints GC to reduce memory.
        """
        from pdf2image import convert_from_path, pdfinfo_from_path  # lazy import

        pdf_path = Path(pdf_path)
        stem = pdf_path.stem

        try:
            info = pdfinfo_from_path(str(pdf_path))
            total_pages = int(info.get("Pages", 0))
        except Exception:
            total_pages = 0

        if total_pages <= 0:
            return

        for page_num in range(1, total_pages + 1):
            try:
                images = convert_from_path(
                    str(pdf_path),
                    dpi=dpi,
                    fmt=fmt.lower(),
                    first_page=page_num,
                    last_page=page_num,
                    thread_count=thread_count,
                )
            except Exception:
                images = []

            if not images:
                continue

            src_img = images[0]
            out_img = src_img.copy()
            page_id = f"{stem}-p{page_num}"

            try:
                yield PageImage(image=out_img, page_id=page_id, variant="full")
            finally:
                try:
                    src_img.close()
                except Exception:
                    pass
                del src_img
                gc.collect()
