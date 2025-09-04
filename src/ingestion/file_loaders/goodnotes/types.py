from __future__ import annotations

from dataclasses import field
from typing import Any, Dict, List, Tuple

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from PIL import Image


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PageImage:
    """Represents a page image with file identity and variant.

    Attributes:
        image: PIL Image for the page.
        filename: Source PDF filename (basename, e.g., "doc.pdf").
        page: 1-based page index in the source PDF.
        variant: Optional variant label (e.g., "full", "01U_grey").
        meta: Arbitrary metadata.
    """

    image: Image.Image
    filename: str
    page: int
    variant: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> Tuple[int, int]:
        """Returns image size as (width, height)."""
        return self.image.size


@dataclass
class DetBox:
    """Detection box consisting of polygon and score.

    Attributes:
        poly: List of points [[x,y], ...].
        score: Detection confidence score.
    """

    poly: List[List[int]]
    score: float


@dataclass
class DetectionResult:
    """Detection result for a page.

    Attributes:
        page: The page image analyzed.
        boxes: List of detected boxes.
    """

    page: PageImage
    boxes: List[DetBox]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CropItem:
    """Represents a cropped region from a page.

    Attributes:
        page: Source page image.
        box: Bounding box (x_min, y_min, x_max, y_max).
        poly_origin: Original polygon from detection.
        det_score: Detection confidence score.
        image: Cropped PIL image.
        expand_ratio: Applied expansion ratio for width/height.
    """

    page: PageImage
    box: Tuple[int, int, int, int]
    poly_origin: List[List[int]]
    det_score: float
    image: Image.Image
    expand_ratio: float = 1.0


@dataclass
class CropsBatch:
    """Collection of crop items resulting from detection."""

    items: List[CropItem]


@dataclass
class RecItem:
    """Recognition result for a single crop.

    Attributes:
        crop: Source crop item.
        text: Recognized text.
        score: Recognition confidence score.
    """

    crop: CropItem
    text: str
    score: float


@dataclass
class RecognitionResult:
    """Collection of recognition results for a batch of crops."""

    items: List[RecItem]


@dataclass
class GroupItem:
    """Represents a group of neighboring text segments.

    Attributes:
        box: Representative rectangle polygon.
        contents: Ordered recognized texts in the group.
        rectangles: Additional rectangle information.
    """

    box: List[List[int]]
    contents: List[str]
    rectangles: Dict[str, Any]


@dataclass
class GroupedCorpus:
    """Structured corpus aggregated from recognition results.

    Attributes:
        content: Joined text from all groups separated by newlines.
        group: List of groups with geometric info and texts.
    """

    content: str
    group: List[GroupItem]
