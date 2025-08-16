# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/01/21 15:57:40
@Author  :  Kevin Wang
@Desc    :  None
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Any

# 你自己定義的 metadata 型別
from objects import DocumentMetadata

# 統一 output 包裝
from pydantic import BaseModel
from typing import Optional, Dict

class LoaderResult(BaseModel):
    content:str  # 放純文本，方便除錯
    metadata:DocumentMetadata
    doc:Any  # 這裡放不同 DocLoader 的內建物件

class DocumentLoader(ABC):
    """文件載入器的抽象基底類別"""

    @abstractmethod
    def _get_metadata(self, path: Union[str, Path]) -> DocumentMetadata:
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> List[LoaderResult]:
        """
        載入文件，回傳 LoaderResult 列表（包含 metadata/doc）
        """
        pass

    def load_multi(self,
                   paths:List[Union[str,Path]],
                   **kwargs,
                   ) -> List[LoaderResult]:
        all_results:List[LoaderResult] = []
        for path in paths:
            all_results.extend(self.load(path))
        return all_results