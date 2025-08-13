# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/01/15 13:44:19
@Author  :  Kevin Wang
@Desc    :  None
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence
from uuid import UUID

from abc import ABC, abstractmethod
from typing import List

from objects import Chunk

from pydantic import BaseModel, Field

class SearchHit(BaseModel):
    chunk:Chunk
    score:float

class VectorIndexStore(ABC):
    """向量索引儲存介面（分片為最小單位）
    """

    @abstractmethod
    def insert(self, chunk:Chunk) -> None:
        """插入新的分片。"""
        pass

    @abstractmethod
    def search(self, query_embedding:Sequence[float], top_k:int=5) -> List[SearchHit]:
        """回傳依相似度降冪排序的搜尋結果。"""
        pass

    @abstractmethod
    def get(self, chunk_id:UUID) -> Optional[Chunk]:
        """取得單一分片，找不到回傳 None。"""
        pass

    @abstractmethod
    def update(self, chunk:Chunk) -> None:
        """更新既有分片；若不存在請拋 NotFoundError。"""
        pass

    @abstractmethod
    def delete(self, chunk_id:UUID) -> None:
        """刪除單一分片。若不存在亦不報錯。"""
        pass

    def upsert(self, chunk:Chunk) -> None:
        """先嘗試更新，不存在時插入"""
        try:
            self.update(chunk)
        except Exception:
            self.insert(chunk)

class LexicalIndexStore(ABC):
    """全文/關鍵字索引儲存介面（分片為最小單位）
    """

    @abstractmethod
    def insert(self, chunk:Chunk) -> None:
        """插入新的分片。"""
        pass

    @abstractmethod
    def search(self, query:str, top_k:int=5) -> List[SearchHit]:
        """以關鍵字/全文檢索，回傳依相關度降冪排序的搜尋結果。"""
        pass

    @abstractmethod
    def get(self, chunk_id:UUID) -> Optional[Chunk]:
        """取得單一分片，找不到回傳 None。"""
        pass

    @abstractmethod
    def update(self, chunk:Chunk) -> None:
        """更新既有分片；若不存在請拋 NotFoundError。"""
        pass

    @abstractmethod
    def delete(self, chunk_id:UUID) -> None:
        """刪除單一分片。若不存在亦不報錯。"""
        pass

    def upsert(self, chunk:Chunk) -> None:
        """先嘗試更新，不存在時插入。"""
        try:
            self.update(chunk)
        except Exception:
            self.insert(chunk)
