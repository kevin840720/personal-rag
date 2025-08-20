# -*- encoding: utf-8 -*-
"""
@File    :  test_pgvector.py
@Time    :  2025/08/14
@Author  :  Kevin Wang
@Desc    :  PGVectorStore 單元測試
"""

import os
from typing import (Generator,
                    List,
                    )
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import text
import pytest
import numpy as np

from conftest import SKIP_PGVECTOR_TESTS
from infra.stores.pgvector import PGVectorStore
from objects import (Chunk,
                     DocumentMetadata,
                     FileType,
                     )

load_dotenv()

@pytest.fixture(scope="function")
def store() -> Generator[PGVectorStore, None, None]:
    """建立測試用的 PGVectorStore 實例"""
    schema_name = f"test_schema_{uuid4()}"  # 每次測試使用不同的 index
    store = PGVectorStore(host=os.getenv("MY_POSTGRE_HOST", "localhost"),
                          port=int(os.getenv("MY_POSTGRE_PORT", "5432")),
                          dbname=os.getenv("MY_POSTGRE_DB_NAME", "postgres"),
                          schema=schema_name,
                          user=os.getenv("MY_POSTGRE_USERNAME", "postgres"),
                          password=os.getenv("MY_POSTGRE_PASSWORD", ""),
                          )
    yield store

    # 刪掉 schema
    with store.engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))

@pytest.fixture
def sample_chunk() -> Chunk:
    """建立測試用的 Chunk"""
    return Chunk(id=uuid4(),
                 content="This is a test document",
                 metadata=DocumentMetadata(file_type=FileType.PDF,
                                           file_name="test.pdf",
                                           is_chunk=True,
                                           chunk_info={"chunk_index": 0, "total_chunks": 1},
                                           ),
                 embedding=[0.1] * 1536,
                 )

@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """建立多個測試用的 Chunk"""
    base_vector = np.eye(1536)
    docs = []
    for idx in range(5):
        embedding = base_vector[idx % 1536].tolist()
        docs.append(Chunk(id=uuid4(),
                          content=f"Chunk document {idx}",
                          metadata=DocumentMetadata(file_type=FileType.PDF,
                                                    file_name="test.pdf",
                                                    is_chunk=True,
                                                    chunk_info={"chunk_index": idx, "total_chunks": 5},),
                          embedding=embedding,
                          ))
    return docs

@pytest.mark.skipif(SKIP_PGVECTOR_TESTS, reason="Skipping PGVector tests")
class TestPGVectorStore:
    def test_add_single_document(self,
                                 store:PGVectorStore,
                                 sample_chunk:Chunk,
                                 ):
        """測試新增單一文件"""
        store.insert(sample_chunk)
        # 驗證搜尋結果
        results = store.search(sample_chunk.embedding, top_k=1)
        assert len(results) == 1
        assert results[0].chunk.id == sample_chunk.id
        assert results[0].chunk.content == sample_chunk.content

    def test_add_multiple_documents(self,
                                    store:PGVectorStore,
                                    sample_chunks:List[Chunk],
                                    ):
        """測試批次新增多個文件"""
        for doc in sample_chunks:
            store.insert(doc)
        results = store.search(sample_chunks[0].embedding, top_k=3)
        assert len(results) == 3
        assert results[0].chunk.id == sample_chunks[0].id
        assert results[0].chunk.content == sample_chunks[0].content

    def test_get_document(self,
                          store:PGVectorStore, 
                          sample_chunk:Chunk,
                          ):
        """測試取得單一分片"""
        store.insert(sample_chunk)
        doc = store.get(sample_chunk.id)
        assert doc is not None
        assert doc.id == sample_chunk.id

    def test_update_document(self,
                             store:PGVectorStore,
                             sample_chunk:Chunk,
                             ):
        """測試更新文件"""
        store.insert(sample_chunk)
        updated_content = "Updated content"
        updated_doc = Chunk(id=sample_chunk.id,
                            content=updated_content,
                            metadata=sample_chunk.metadata,
                            embedding=sample_chunk.embedding,
                            )
        store.update(updated_doc)
        results = store.search(sample_chunk.embedding, top_k=1)
        assert results[0].chunk.content == updated_content

    def test_update_nonexistent_document(self,
                                         store:PGVectorStore,
                                         sample_chunk:Chunk,
                                         ):
        """測試更新不存在的文件"""
        with pytest.raises(ValueError):
            store.update(sample_chunk)

    def test_delete_document(self,
                             store:PGVectorStore,
                             sample_chunk:Chunk,
                             ):
        """測試刪除文件"""
        store.insert(sample_chunk)
        store.delete(sample_chunk.id)
        doc = store.get(sample_chunk.id)
        assert doc is None

    def test_search_empty_store(self,
                                store:PGVectorStore,
                                ):
        """測試在空的 store 中搜尋"""
        results = store.search([0.1]*1536, top_k=5)
        assert len(results) == 0

    def test_upsert_document(self,
                             store:PGVectorStore,
                             sample_chunk:Chunk,
                             ):
        """測試 upsert 文件行為"""
        store.upsert(sample_chunk)
        assert store.get(sample_chunk.id) is not None

        updated_content = "Updated content"
        updated_doc = Chunk(id=sample_chunk.id,
                            content=updated_content,
                            metadata=sample_chunk.metadata,
                            embedding=sample_chunk.embedding,
                            )
        store.upsert(updated_doc)
        results = store.search(sample_chunk.embedding, top_k=1)
        assert results[0].chunk.content == updated_content

    def test_upsert_multiple_documents(self,
                                       store:PGVectorStore,
                                       sample_chunks:list,
                                       ):
        """測試 upsert 多個文件"""
        for doc in sample_chunks:
            store.upsert(doc)
        for doc in sample_chunks:
            assert store.get(doc.id) is not None

    def test_search_with_invalid_dimension(self,
                                           store:PGVectorStore,
                                           sample_chunk:Chunk,
                                           ):
        """測試使用錯誤維度的向量搜尋"""
        store.insert(sample_chunk)
        with pytest.raises(Exception):  # 可根據實際情況改用更精確的 exception
            store.search([0.1]*10, top_k=5)
