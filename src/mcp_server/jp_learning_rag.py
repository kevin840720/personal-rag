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
from mcp.server.fastmcp.prompts import Prompt
from mcp.types import TextContent

from cache.redis import RedisCacheHandler
from embedding.openai_embed import OpenAIEmbeddingModel
from infra.stores.pgvector import PGVectorStore
from infra.stores.elasticsearch import ElasticsearchBM25Store
from infra.stores.base import VectorIndexStore, SearchHit
from objects import Chunk

load_dotenv()

class JapaneseLearningRAGServer:
    def __init__(self, host, port):
        # 將系統提示直接設為伺服器 instructions，供通用 orchestrator 使用
        self.mcp = FastMCP("JapaneseLearningRAG",
                           instructions=self.system_prompt(),
                           log_level='DEBUG',
                           # 下面三個參數是 streamable-http mode 才會使用，連線位置是 http://localhost:56481/mcp
                           host=host,
                           port=port,
                           streamable_http_path="/mcp",
                           )

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

    def system_prompt(self) -> str:  
        return """
        你是一個活潑開朗的在台日文教師。你的工作是提供以簡單易懂的方式講述有關日文文法、文化等內容。
        使用 search_japanese_note 工具時，參數 keywords 必填，提供 2–6 個中/日文關鍵詞；

        行為守則
        - 如果你的回答中包含「日文學習筆記」的資訊，必須在回答的最後回傳引用原文的片段與對應頁碼。
        - 使用 search_japanese_note 工具時，參數 keywords 必填，提供 2–6 個中/日文關鍵詞；例如，當使用者詢問奧運時，「奧運」、「オリンピック」都要出現在 keywords。
        - 有關「日文學習筆記」的 metadata: 頁碼、來源、位置等僅取自 metadata，不得從 content/OCR 推斷。
        - 語言一致: 用戶使用何種語言即以相同語言回覆；引文原語可保留但加註來源。
        - 承認錯誤：如果工具搜尋失敗，回傳錯誤訊息，告知使用者失敗的原因
        """
    
    async def search_japanese_note(
        self,
        query: Annotated[str, Field(description="直接用使用者的問題進行語意搜尋")],
        keywords: Annotated[list[str], Field(description="用關鍵字搜尋，建議 2～6 個單詞，要翻譯成中文與日文，例如：「奧運」和「オリンピック」要同時出現")],
        top_embedding_k: Annotated[int, Field(default=3, description="Embedding 檢索數量，只會使用 `question`")] = 3,
        top_keyword_k: Annotated[int, Field(default=3, description="每個關鍵字 BM25 檢索數量，只會使用 `keywords`")] = 3,
    ) -> TextContent:
        """
        用途：
            從使用者的日文學習筆記中搜尋並擷取相關段落，以回答提問或提供參考內容。

        使用時機：
            - 查詢偏長、敘述性強，且需要語意層次理解（如目的、定義、流程、說明）時。
            - 帶有明確關鍵字，想要用傳統方式搜尋時。
            - 需要從多篇筆記彙整重點或比對相近概念時。

        限制：
            - 僅檢索已索引的筆記內容；若無結果，可能是尚未收錄或關鍵詞不匹配。
            - 如果你打算搜尋資訊，你的 keyword 要翻譯成中文與日文，例如：使用者搜尋「奧運」，你在時要用「奧運」和「オリンピック」。
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

        merged:List[Chunk] = list(unique.values())

        lines = [
            f"找到 {len(merged)} 個相關文件：",
            # "提示：頁碼與來源請一律依據 metadata，不得從內容/OCR 推斷。",
            # "     必須回傳引用原文。"
        ]
        for i, ch in enumerate(merged, 1):
            content = ch.content
            meta = ch.metadata
            lines.append(f"{meta.file_name}\n文件資訊: {meta}\n內文: {content}\n")

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
#             question="日文的助詞是什麼？",
#             keywords=["助詞", "文法"],
#             top_embedding_k=2,
#             top_keyword_k=2,
#         )
#         print(result.text)

#     asyncio.run(main())
