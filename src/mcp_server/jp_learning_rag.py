# -*- encoding: utf-8 -*-
"""
@File    :  rag_tools.py
@Time    :  2025/08/23 19:30:01
@Author  :  Kevin Wang
@Desc    :  None
"""

import os
import asyncio
from typing import (Annotated,
                    List,
                    )

from dotenv import load_dotenv
from pydantic import Field

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from cache.redis import RedisCacheHandler
from embedding.openai_embed import OpenAIEmbeddingModel
from infra.stores.pgvector import PGVectorStore
from infra.stores.elasticsearch import ElasticsearchBM25Store
from objects import Chunk
from monitoring.langfuse_client import get_langfuse_client, observe_if_enabled

load_dotenv()

class JapaneseLearningRAGServer:
    def __init__(self,
                 *,
                 host:str="127.0.0.1",
                 port:int=8000,
                 ):
        # 將系統提示直接設為伺服器 instructions，供通用 orchestrator 使用
        self.mcp = FastMCP("JapaneseLearningRAG",
                           instructions="私人日文筆記查詢系統，查詢有關日文單字、文法、筆記等相關問題",
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
        self.mcp.add_tool(self.japanese_note_reply_instruction)
        self.mcp.add_tool(self.search_japanese_note)

    @observe_if_enabled(name="mcp.JapaneseLearningRAGServer.retrieve_chunks",
                        capture_input=True,
                        capture_output=True,
                        )
    async def retrieve_chunks(self,
                               *,
                               query:str,
                               keywords:List[str],
                               top_embedding_k:int=3,
                               top_keyword_k:int=3,
                               ) -> List[Chunk]:
        """核心檢索：回傳合併去重後的 Chunk 清單。

        - 不做字串渲染。
        - 不做相容性層與吞例外；發生錯誤讓它拋出即可。
        """
        keywords = keywords or []

        async def embed_search(q:str):
            qv = self.embedder.encode(q)
            return self.vec_store.search(qv, top_k=top_embedding_k)

        async def bm25_search(kw:str):
            return self.lex_store.search(kw, top_k=top_keyword_k)

        tasks = []
        if query:
            tasks += [embed_search(query)]
        if keywords:
            tasks += [bm25_search(kw) for kw in keywords]

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)

        all_chunks:List[Chunk] = [hit.chunk for hits in results for hit in hits]
        if not all_chunks:
            return []

        unique = {}
        for ch in all_chunks:
            cid = ch.id
            if cid and cid not in unique:
                unique[cid] = ch

        return list(unique.values())

    @observe_if_enabled(name="mcp.JapaneseLearningRAGServer.japanese_note_reply_instruction",
                        capture_input=False,
                        capture_output=True,
                        )
    def japanese_note_reply_instruction(self) -> str: 
        """
        工具用途：
            - 當引用「日文學習筆記」的內容來生成回覆時使用。

        行為規範：
            - 回覆必須以 Markdown 輸出，最多會有四個部份：
                0. 前言：簡短回答
                1. 筆記：先對筆記做出彙整回答，接著引用筆記的原文，並在最後標註來源（檔名、頁碼或段落）。若找不到，請填寫「查無資料」。
                2. 網路資訊（可選）：在網路搜尋的結果，包含教學網站、部落格等等，需要在引用的段落後標註來源（連結）。
                3. 總整理：如果資料來源超過三處，在最後一段進行彙整。
            - 不得自行編造文件條文或規範。
            - 若完全沒有檢索結果，回覆「未找到相關文件」。
        """
        instructions = TextContent(
            type="text",
            text=("（回覆必須以 Markdown 輸出，架構如下：）"
                  "（簡短回答）\n"
                  "## 筆記\n"
                  "### 摘錄\n"
                  "（對筆記做出彙整回答）\n\n"
                  "### 筆記原文\n"
                  "（引用筆記的原文，並在最後標註來源，若無資料，則填寫「查無資料」）\n\n"
                  "## 網路資訊\n"
                  "（在網路搜尋的結果，如果未搜尋，則移除此段）\n\n"
                  "## 總整理\n"
                  "（如果資料來源超過三處，在最後一段進行彙整，否則移除此段）\n"
                  )
        )
        return instructions

    @observe_if_enabled(name="mcp.JapaneseLearningRAGServer.search_japanese_note",
                        capture_input=True,
                        capture_output=True,
                        )
    async def search_japanese_note(
        self,
        query:Annotated[str, Field(description="直接用使用者的問題進行語意搜尋")],
        keywords:Annotated[list[str], Field(description="用關鍵字搜尋，建議 2～6 個單詞，要翻譯成中文與日文，例如：「奧運」和「オリンピック」要同時出現")],
        top_embedding_k:Annotated[int, Field(default=3, description="Embedding 檢索數量，只會使用 `question`")]=3,
        top_keyword_k:Annotated[int, Field(default=3, description="每個關鍵字 BM25 檢索數量，只會使用 `keywords`")]=3,
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
        client = get_langfuse_client()
        chunks:List[Chunk]
        if client is not None:
            with client.start_as_current_span(name="retrieve_chunks",
                                              input={'query': query,
                                                     'keywords': keywords,
                                                     'top_embedding_k': top_embedding_k,
                                                     'top_keyword_k': top_keyword_k,
                                                     },
                                              ) as span:
                chunks = await self.retrieve_chunks(query=query,
                                                     keywords=keywords,
                                                     top_embedding_k=top_embedding_k,
                                                     top_keyword_k=top_keyword_k,
                                                     )
                span.update(output={'chunk_count': len(chunks),
                                    'chunk_ids': [str(ch.id) for ch in chunks],
                                    })
        else:
            chunks = await self.retrieve_chunks(query=query,
                                                 keywords=keywords,
                                                 top_embedding_k=top_embedding_k,
                                                 top_keyword_k=top_keyword_k,
                                                 )

        if not chunks:
            return TextContent(type="text", text="未找到相關文件。")

        lines = [
            f"找到 {len(chunks)} 個相關文件：",
        ]
        for i, ch in enumerate(chunks, 1):
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
