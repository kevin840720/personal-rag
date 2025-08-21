# -*- encoding: utf-8 -*-
"""
@File    :  openai.py
@Time    :  2025/08/20 21:48:59
@Author  :  Kevin Wang
@Desc    :  None
"""

import hashlib
from abc import ABC
from typing import (Any,
                    List,
                    Literal,
                    Optional,
                    Sequence,
                    )
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from openai import OpenAI
import numpy as np

from cache.base import BaseCacheHandler
from embedding.base import EmbeddingModel

class EmbeddingError(Exception):
    pass

class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI Embedding 模型實作"""

    SUPPORTED_MODELS = ("text-embedding-3-small", "text-embedding-3-large")

    def __init__(self,
                 api_key:str,
                 model_name:Literal["text-embedding-3-small","text-embedding-3-large"]="text-embedding-3-small",
                 max_retries:int=3,
                 max_workers:Optional[int]=None,
                 memory_cache:Optional[BaseCacheHandler]=None,
                 ):
        """
        初始化 OpenAI Embedding 模型。

        Args:
            api_key (str): OpenAI API 金鑰。
            model_name (str, optional): OpenAI embedding 模型名稱。
            max_retries (int, optional): 最大重試次數。
            max_workers (Optional[int], optional): 執行緒數，預設 CPU 核心數減 2。
            memory_cache (Optional[BaseCacheHandler], optional): 快取處理器。
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        if model_name not in self.SUPPORTED_MODELS:
            raise EmbeddingError(f"model_name 必須為 {self.SUPPORTED_MODELS} 之一，收到：{model_name}")

        self.max_retries = max_retries
        self.max_workers = max_workers or max(cpu_count()-2, 1)
        self.memory_cache = memory_cache

    def _get_cache_key(self, text:str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self, text:str) -> Optional[List[float]]:
        if not self.memory_cache:
            return None
        cache_key = self._get_cache_key(text)
        return self.memory_cache.get(cache_key)

    def _save_cache(self,
                    text:str,
                    embedding:List[float],
                    ):
        if not self.memory_cache:
            return
        cache_key = self._get_cache_key(text)
        self.memory_cache.set(cache_key, embedding)

    def encode(self, text:str) -> List[float]:  # 官方說明：一次 API 請求不能超過 8192 token
        if not text:
            raise EmbeddingError("Input text cannot be empty.")
        cached_embedding = self._load_cache(text)
        if cached_embedding is not None:
            return cached_embedding

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(input=text,
                                                         model=self.model_name,
                                                         )
                embedding = response.data[0].embedding
                self._save_cache(text, embedding)
                return embedding
            except Exception as err:
                if attempt == self.max_retries-1:
                    raise EmbeddingError(f"Embedding generation failed after {self.max_retries} attempts") from err

    def encode_batch(self,
                     texts:Sequence[str],
                     ) -> List[List[float]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.encode, texts))

    @property
    def dim(self) -> int:
        sample_vec = self.encode("dimension probe")
        return len(sample_vec)

    @property
    def name(self) -> str:
        return self.model_name

    def similarity(self,
                   text1:str,
                   text2:str,
                   ) -> float:
        vec1 = np.array(self.encode(text1))
        vec2 = np.array(self.encode(text2))
        return float(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    def most_similar(self,
                     query:str,
                     candidates:List[str],
                     top_k:int=1,
                     ) -> List[tuple[str,float]]:
        """cos similarity"""
        query_vec = np.array(self.encode(query))
        sims = []
        for text in candidates:
            vec = np.array(self.encode(text))
            score = float(np.dot(query_vec, vec)/(np.linalg.norm(query_vec)*np.linalg.norm(vec)))
            sims.append((text, score))
        return sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
