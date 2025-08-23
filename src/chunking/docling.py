# -*- encoding: utf-8 -*-
"""
@File    :  docling.py
@Time    :  2025/08/17 16:31:42
@Author  :  Kevin Wang
@Desc    :  None
"""

from copy import deepcopy
from typing import (List,
                    Optional,
                    )
import uuid

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (ChunkingDocSerializer,
                                                                  ChunkingSerializerProvider,
                                                                  )
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DoclingDocument
from pydantic import ConfigDict
import tiktoken

from chunking.base import ChunkProcessor
from ingestion.utils import JSONTableSerializer
from objects import (Chunk,
                     DocumentMetadata,
                     )
class OpenAITokenizer(BaseTokenizer):
    """其實是照抄 docling_core.transforms.chunker.tokenizer.openai.OpenAITokenizer
    但不希望 docling 與 openai 耦合度太高
    所以拉出來寫"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: tiktoken.Encoding
    max_tokens: int

    def count_tokens(self, text:str) -> int:
        return len(self.tokenizer.encode(text=text))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> tiktoken.Encoding:
        return self.tokenizer

class JSONTableSerializerProvider(ChunkingSerializerProvider):
    """JSON-specific table item serializer provider for Docling HybridChunker

    可以參考 https://docling-project.github.io/docling/examples/advanced_chunking_and_serialization/#configuring-a-different-strategy
    """
    def get_serializer(self, doc:DoclingDocument):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=JSONTableSerializer(),  # configuring a different table serializer
        )

class DoclingChunkProcessor(ChunkProcessor):
    """
    ✅ Docling 分割邏輯總結

    1. **先按層級切（HierarchicalChunker）**  
    - **目的**：保留文件原始的語意與結構，例如段落（paragraphs）、表格、標題、圖片等。  
    - **程式位置**：`self._inner_chunker.chunk(...)`  
    - **依據**：文件中 `doc_items` 與 `headings` 等 metadata。

    2. **再按文本長度切（token-aware）**  
    - **目的**：確保每個 chunk 不超過 tokenizer 支援的 `max_tokens`。  
    - **步驟**：  
        - **(a)** `_split_by_doc_items()`：粗略從結構上切成幾塊，每塊不超過 token 限制（用 `doc_item` 作單位）。  
        - **(b)** `_split_using_plain_text()`：若仍過長，則用 OpenAI tokenizer 對文字本身做細切（透過 `sem_chunker`）。  
    - **關鍵點**：非純粹字數，而是按 tokenizer 切的 token 數（e.g., BPE 或 Tiktoken 模型）。

    3. **最後合併過小 chunk（optional）**  
    - **條件**：  
        - `merge_peers=True`  
        - 合併後仍在 `max_tokens` 範圍內  
        - 且 metadata（尤其是 `headings`）一致  
    - **目標**：減少碎片化輸出，使上下文更完整。  
    - **程式位置**：`_merge_chunks_with_matching_metadata(...)`

    """
    def __init__(self,
                 tokenizer:Optional[OpenAITokenizer]=None,  # OpenAITokenizer 是 Docling 的套件提供的
                 merge_peers:bool=True,
                 **kwargs,
                 ):
        """Initialize the Docling chunk processor.
        
        Args:
            tokenizer: Tokenizer instance for token-aware chunking
            merge_peers: Whether to merge adjacent chunks with similar metadata
            **kwargs: Additional arguments for the HybridChunker
        """
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=8*1024,
        ) if tokenizer is None else tokenizer
        self._chunker = HybridChunker(tokenizer=self.tokenizer,
                                      merge_peers=merge_peers,
                                      serializer_provider=JSONTableSerializerProvider(),
                                      **kwargs,
                                      )
        # self._chunks:List[DocChunk] = []  # FIXME: 注意需不需要

    def process(self,
                doc:DoclingDocument,
                metadata:DocumentMetadata,
                ) -> List[Chunk]:
        """Process the input document into chunks.
        
        Args:
            doc: Document to be processed
            
        Returns:
            List[Chunk]: List of text chunks
        """
        chunks = list(self._chunker.chunk(dl_doc=doc))
        total_chunks = len(chunks)
        chunk_docs = []
        for idx, chunk_obj in enumerate(chunks):
            chunk_meta = deepcopy(metadata)
            chunk_meta.is_chunk = True
            chunk_meta.chunk_info = {'chunk_index': idx,
                                     'total_chunks': total_chunks,
                                     'chunk_size': len(chunk_obj.text),
                                     'chunk_tokens': self.tokenizer.count_tokens(chunk_obj.text),
                                     }
            chunk_doc = Chunk(id=uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_obj.model_dump_json())+str(chunk_meta.model_dump_json())),
                              content=chunk_obj.text,
                              metadata=chunk_meta,
                              _raw_chunk=chunk_obj,
                              )
            chunk_doc._raw_chunk=chunk_obj
            chunk_docs.append(chunk_doc)
        return chunk_docs
