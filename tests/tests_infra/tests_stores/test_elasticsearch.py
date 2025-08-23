# -*- encoding: utf-8 -*-
"""
@File    :  test_elasticsearch.py
@Time    :  2025/08/13 20:55:14
@Author  :  Kevin Wang
@Desc    :
"""

from typing import (Generator,
                    List,
                    )
from uuid import uuid4
import os

from dotenv import load_dotenv
import numpy as np
import pytest

from conftest import SKIP_ELASTICSEARCH_TESTS
from infra.stores.elasticsearch import ElasticsearchBM25Store
from infra.stores.errors import ElasticsearchActionError
from objects import (Chunk,
                     DocumentMetadata,
                     FileType,
                     )

load_dotenv()

@pytest.fixture
def store() -> Generator[ElasticsearchBM25Store, None, None]:
    """建立測試用的 ElasticsearchBM25Store 實例"""
    store = ElasticsearchBM25Store(host=os.getenv("MY_ELASTIC_HOST", "localhost"),
                                   port=int(os.getenv("MY_ELASTIC_PORT", "9200")),
                                   index_name=f"test_index_{uuid4()}",  # 每次測試使用不同的 index
                                   username=os.getenv("MY_ELASTIC_USERNAME"),
                                   password=os.getenv("MY_ELASTIC_PASSWORD"),
                                   ssl_verify=False,
                                   )
    yield store
    # 清理：刪除測試用的 index
    store.clean()

@pytest.fixture
def sample_document() -> Chunk:
    """建立測試用的文件"""
    return Chunk(id=uuid4(),
                 content="This is a test document",
                 metadata=DocumentMetadata(file_type=FileType.PDF,
                                           file_name="test.pdf",
                                           is_chunk=True,
                                           chunk_info={"chunk_index": 0,
                                                       'total_chunks': 1,
                                                       },
                                           ),
                 embedding=[0.1]*1536,
                 )

@pytest.fixture
def sample_documents() -> List[Chunk]:
    """建立多個測試用的文件"""
    base_vector = np.eye(1536)  # 創建一個單位矩陣，每行是正交向量
    documents = []
    for idx in range(10):
        embedding = base_vector[idx % 1536].tolist()  # 選擇不同的正交向量作為 embedding
        documents.append(Chunk(id=uuid4(),
                               content=f"Test document {idx}",
                               metadata=DocumentMetadata(file_type=FileType.PDF,
                                                         file_name="test.pdf",
                                                         is_chunk=True,
                                                         chunk_info={"chunk_index": idx,
                                                                     'total_chunks': 10,
                                                                     },
                                                         ),
                               embedding=embedding,
                               ))
    return documents

@pytest.mark.skipif(SKIP_ELASTICSEARCH_TESTS, reason="Skipping Elasticsearch tests")
class TestElasticsearchBM25Store:
    def test_add_single_document(self,
                                 store:ElasticsearchBM25Store,
                                 sample_document:Chunk,
                                 ):
        """測試新增單一文件"""
        store.insert(sample_document)
        assert len(store) == 1
        # 驗證搜尋結果
        results = store.search_by_vector_similarity(sample_document.embedding, top_k=1)
        assert len(results) == 1
        assert str(results[0].chunk.id) == str(sample_document.id)
        assert results[0].chunk.content == sample_document.content

    def test_add_multiple_documents(self,
                                    store:ElasticsearchBM25Store,
                                    sample_documents:List[Chunk],
                                    ):
        """測試批次新增多個文件"""
        for doc in sample_documents:
            store.insert(doc)
        assert len(store) == len(sample_documents)
        # 驗證搜尋結果
        results = store.search_by_vector_similarity(sample_documents[0].embedding, top_k=3)
        assert len(results) == 3
        assert str(results[0].chunk.id) == str(sample_documents[0].id)
        assert results[0].chunk.content == sample_documents[0].content

    def test_add_id_conflict(self,
                             store:ElasticsearchBM25Store,
                             sample_document:Chunk,
                             ):
        """測試 ID 已存在時觸發 ElasticsearchActionError"""
        store.insert(sample_document)
        assert len(store) == 1

        # 嘗試插入相同的文件，應該觸發 ElasticsearchActionError
        with pytest.raises(ElasticsearchActionError):
            store.insert(sample_document)

    def test_search_empty_store(self,
                                store:ElasticsearchBM25Store,
                                ):
        """測試在空的 store 中搜尋"""
        assert len(store) == 0
        results = store.search("Test", top_k=5)
        assert len(results) == 0

    def test_search_with_metadata_filter(self,
                                         store:ElasticsearchBM25Store,
                                         sample_documents:List[Chunk],
                                         ):
        """測試帶有過濾條件的搜尋（metadata）"""
        for doc in sample_documents:
            store.insert(doc)

        # 使用 metadata 過濾
        results = store.search(
            query_text="Test",
            top_k=3,
            metadata_filter={"term": {"metadata.chunk_info.chunk_index": 2}}
        )
        assert len(results) == 1
        assert results[0].chunk.metadata.chunk_info["chunk_index"] == 2

    def test_search_by_vector_empty_store(self,
                                          store:ElasticsearchBM25Store,
                                          ):
        """測試在空的 store 中搜尋"""
        assert len(store) == 0
        results = store.search_by_vector_similarity([0.1]*1536, top_k=5)
        assert len(results) == 0

    def test_search_by_vector_with_metadata_filter(self,
                                                   store:ElasticsearchBM25Store,
                                                   sample_documents:List[Chunk],
                                                   ):
        """測試帶有過濾條件的搜尋（metadata）"""
        for doc in sample_documents:
            store.insert(doc)

        # 使用 metadata 過濾
        results = store.search_by_vector_similarity(
            query_embedding=sample_documents[0].embedding,
            top_k=3,
            metadata_filter={"term": {"metadata.chunk_info.chunk_index": 2}}
        )
        assert len(results) == 1
        assert results[0].chunk.metadata.chunk_info["chunk_index"] == 2

    def test_update_document(self,
                             store:ElasticsearchBM25Store,
                             sample_document:Chunk,
                             ):
        """測試更新文件"""
        # 添加原始文件
        store.insert(sample_document)

        # 更新文件
        updated_content = "Updated content"
        updated_doc = Chunk(id=sample_document.id,
                              content=updated_content,
                              metadata=sample_document.metadata,
                              embedding=sample_document.embedding,
                              )
        store.update(updated_doc)

        # 驗證更新
        results = store.search_by_vector_similarity(sample_document.embedding, top_k=1)
        assert results[0].chunk.content == updated_content

    def test_update_nonexistent_document(self,
                                         store:ElasticsearchBM25Store,
                                         sample_document:Chunk,
                                         ):
        """測試更新不存在的文件"""
        assert len(store) == 0
        with pytest.raises(ElasticsearchActionError):
            store.update(sample_document)

    def test_delete_document(self,
                             store:ElasticsearchBM25Store,
                             sample_document:Chunk,
                             ):
        """測試刪除文件"""
        store.insert(sample_document)
        assert len(store) == 1
        store.delete(sample_document.id)
        assert len(store) == 0

    def test_get_document(self,
                          store:ElasticsearchBM25Store,
                          sample_document:Chunk):
        """測試獲取特定文件"""
        store.insert(sample_document)
        results = store.get(sample_document.id)
        assert len(results) == 1
        assert str(results[0].id) == str(sample_document.id)
        assert results[0].content == sample_document.content

    def test_get_nonexistent_document(self,
                                      store:ElasticsearchBM25Store,
                                      ):
        """測試獲取不存在的文件"""
        with pytest.raises(ElasticsearchActionError):
            store.get(uuid4())

    def test_upsert_document(self,
                             store:ElasticsearchBM25Store,
                             sample_document:Chunk,
                             ):
        """測試 upsert 文件行為"""
        # 初次 upsert 應新增文件
        store.upsert(sample_document)
        assert len(store) == 1

        # 更新同一 ID 的文件
        updated_content = "Updated content"
        updated_doc = Chunk(id=sample_document.id,
                              content=updated_content,
                              metadata=sample_document.metadata,
                              embedding=sample_document.embedding,
                              )
        store.upsert(updated_doc)

        # 確認更新成功
        assert len(store) == 1
        results = store.search_by_vector_similarity(sample_document.embedding, top_k=1)
        assert str(results[0].chunk.id) == str(sample_document.id)
        assert results[0].chunk.content == updated_content

    def test_upsert_multiple_documents(self,
                                       store:ElasticsearchBM25Store,
                                       sample_documents:List[Chunk],
                                       ):
        """測試 upsert 多個文件"""
        # 初次插入文件
        for doc in sample_documents:
            store.upsert(doc)
        assert len(store) == len(sample_documents)  # 確認所有文件插入成功

        # 更新其中部分文件
        updated_content = "Updated document"
        updated_doc = Chunk(id=sample_documents[0].id,
                              content=updated_content,
                              metadata=sample_documents[0].metadata,
                              embedding=sample_documents[0].embedding,
                              )
        store.upsert(updated_doc)

        # 確認更新成功
        results = store.search_by_vector_similarity(sample_documents[0].embedding, top_k=1)
        assert results[0].chunk.content == updated_content
        assert len(store) == len(sample_documents)  # 文件數量應保持不變

    def test_clean_specific_documents(self,
                                      store:ElasticsearchBM25Store,
                                      sample_documents:List[Chunk],
                                      ):
        """測試清理特定條件的文件"""
        for doc in sample_documents:
            store.insert(doc)

        # 清理特定條件的文件
        store.clean({"term": {"metadata.chunk_info.chunk_index": 2}})

        # 驗證結果
        assert len(store) == len(sample_documents) - 1
        results = store.search_by_vector_similarity(query_embedding=sample_documents[0].embedding,
                                                    top_k=10,
                                                    metadata_filter={"term": {"metadata.chunk_index": 2}},
                                                    )
        assert len(results) == 0

    def test_clean_all_documents(self,
                                 store:ElasticsearchBM25Store,
                                 sample_documents:List[Chunk],
                                 ):
        """測試清理所有文件"""
        for doc in sample_documents:
            store.insert(doc)
        assert len(store) == len(sample_documents)

        # 清理所有文件
        store.clean()
        assert len(store) == 0
