# -*- encoding: utf-8 -*-
"""
@File    :  rag_tools.py
@Time    :  2025/08/23 19:30:01
@Author  :  Kevin Wang
@Desc    :  None
"""

import os
import asyncio
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from pydantic import Field

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from cache.redis import RedisCacheHandler
from embedding.openai_embed import OpenAIEmbeddingModel
from infra.stores.pgvector import PGVectorStore
from infra.stores.elasticsearch import ElasticsearchBM25Store
from infra.stores.base import VectorIndexStore, SearchHit
from objects import Chunk

load_dotenv()

class JapaneseLearningRAGServer:
    def __init__(self):
        self.mcp = FastMCP("JapaneseLearningRAG")

        # 初始化依賴
        self.embedder = OpenAIEmbeddingModel(
            api_key=os.getenv("OPEN_AI_API"),
            model_name="text-embedding-3-small",
            memory_cache=RedisCacheHandler(
                host=os.getenv("MY_REDIS_HOST"),
                port=os.getenv("MY_REDIS_PORT"),
                password=os.getenv("MY_REDIS_PASSWORD"),
            ),
        )
        self.vec_store = PGVectorStore(
            host=os.getenv("MY_POSTGRE_HOST"),
            port=os.getenv("MY_POSTGRE_PORT"),
            dbname=os.getenv("MY_POSTGRE_DB_NAME"),
            schema="Japanese-Learning",
            user=os.getenv("MY_POSTGRE_USERNAME"),
            password=os.getenv("MY_POSTGRE_PASSWORD"),
        )
        self.lex_store = ElasticsearchBM25Store(
            host=os.getenv("MY_ELASTIC_HOST"),
            port=os.getenv("MY_ELASTIC_PORT"),
            index_name="japanese-learning",
            username=os.getenv("MY_ELASTIC_USERNAME"),
            password=os.getenv("MY_ELASTIC_PASSWORD"),
        )

        # 註冊工具
        self.mcp.add_tool(self.search_japanese_note)

    async def search_japanese_note(
        self,
        query: Annotated[str, Field(..., description="使用者查詢問題")],
        keywords: Annotated[
            Optional[List[str]],
            Field(default=None, description="輔助關鍵字 (0~3 個)，用於 BM25 檢索"),
        ] = None,
        top_embedding_k: Annotated[int, Field(default=3, description="Embedding 檢索數量")] = 3,
        top_keyword_k: Annotated[int, Field(default=3, description="每個關鍵字 BM25 檢索數量")] = 3,
    ) -> TextContent:
        """
        用途：
            從使用者的日文學習筆記中搜尋並擷取相關段落，以回答提問或提供參考內容。

        使用時機：
            - 查詢偏長、敘述性強，且需要語意層次理解（如目的、定義、流程、說明）時。
            - 無明確關鍵字，僅能以概念或背景描述需求時。
            - 需要從多篇筆記彙整重點或比對相近概念時。

        限制：
            - 僅檢索已索引的筆記內容；若無結果，可能是尚未收錄或關鍵詞不匹配。
        """
        keywords = keywords or []

        async def embed_search() -> SearchHit:
            qv = self.embedder.encode(query)
            return self.vec_store.search(qv, top_k=top_embedding_k)

        async def bm25_search(kw:str) -> SearchHit:
            return self.lex_store.search(kw, top_k=top_keyword_k)


        tasks = [embed_search()] + [bm25_search(kw) for kw in keywords]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合併 + 去重
        all_chunks:List[Chunk] = [hit.chunk for hits in results for hit in hits]

        if not all_chunks:
            return TextContent(type="text", text="未找到相關文件。")

        unique = {}
        for ch in all_chunks:
            cid = ch.id
            if cid and cid not in unique:
                unique[cid] = ch

        merged = list(unique.values())

        lines = [f"找到 {len(merged)} 個相關文件："]
        for i, ch in enumerate(merged, 1):
            content = ch.content
            meta = ch.metadata
            lines.append(f"文件{i} 文件資訊: {meta}\n內文: {content}\n")

        return TextContent(type="text", text="\n".join(lines))

    def run(self, transport="stdio"):
        self.mcp.run(transport=transport)


if __name__ == "__main__":
    server = JapaneseLearningRAGServer()
    server.run("stdio")

# if __name__ == "__main__":
#     async def main():
#         server = JapaneseLearningRAGServer()
#         result = await server.search_japanese_note(
#             query="日文的助詞是什麼？",
#             keywords=["助詞", "文法"],
#             top_embedding_k=2,
#             top_keyword_k=2,
#         )
#         print(result.text)

#     asyncio.run(main())