# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/08/20 21:40:53
@Author  :  Kevin Wang
@Desc    :  None
"""

from abc import ABC, abstractmethod
from typing import List, Sequence

class EmbeddingModel(ABC):
    """RAG 系統 Embedding 模型的抽象介面。"""

    @abstractmethod
    def encode(self,
               text:str,
               ) -> Sequence[float]:
        """將單一句子轉換為向量表示。

        Args:
            text (str): 欲轉換的文本。

        Returns:
            Union[List[float], Any]: 對應的向量（可自定義回傳型態，如 np.ndarray）。
        """
        pass

    @abstractmethod
    def encode_batch(self,
                     texts:Sequence[str],
                     ) -> Sequence[Sequence[float]]:
        """批次將多個句子轉換為向量。

        Args:
            texts (Sequence[str]): 多個文本。

        Returns:
            Union[List[List[float]], Any]: 多個向量。
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量維度。

        Returns:
            int: 向量維度。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名稱。

        Returns:
            str: 模型名稱。
        """
        pass
