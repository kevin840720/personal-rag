# -*- encoding: utf-8 -*-
"""
@File    :  elasticsearch.py
@Time    :  2025/08/13 20:55:14
@Author  :  Kevin Wang
@Desc    :  None
@Note    :  SQLAlchemy 在 Elasticsearch 只支援 read-only (需安裝外部 dialects，參考 https://docs.sqlalchemy.org/en/14/dialects/)
@Note    :  BM25 是一種用於評估查詢詞與文件之間的相關性的演算法，廣泛應用於搜尋引擎和文件檢索系統。
            其基於詞頻（TF, Term Frequency）與逆向文件頻率（IDF, Inverse Document Frequency），
            同時考慮文件長度對詞頻的影響，使其相較於傳統 TF-IDF 模型更加精確。
                - 考慮了文件長度的歸一化，使不同長度的文件在排名上更公平。
                - 可調整參數 k1 和 b 來適應不同應用場景。
                - 相較於 TF-IDF，更能有效提升搜尋結果的相關性。

            BM25 分數計算如下：
                                     IDF(qi) × ( tf(qi, D) × (k1 + 1) )  
                BM25(qi, D) =  ————————————————————————————————————————————————
                                tf(qi, D) + k1 × (1 - b + b × (|D| / avgDL) )  

                    其中：
                        - qi 表示查詢中的詞項（term）
                        - D 表示文件
                        - tf(qi, D) 是詞項 qi 在文件 D 中的詞頻，計算方式如下：

                                               出現在文件 D 中的詞項 qi 的次數
                                tf(qi, D) =  ————————————————————————————————
                                                     該文件中的總詞數

                        - |D| 是文件 D 的長度（即詞的總數）
                        - avgDL 是整個文件集合中的平均文件長度
                        - k1 和 b 為調整參數
                        - IDF(qi) 是該詞的逆向文件頻率，計算方式如下：

                                              ⎛   N - df(qi) + 0.5       ⎞
                                IDF(qi) = log ⎜  ——————————————————  + 1 ⎟
                                              ⎝      df(qi) + 0.5        ⎠
                                    其中：
                                    - N 為文件總數
                                    - df(qi) 為包含詞項 qi 的文件數量

            參數說明：
                - **k1**（通常取值範圍為 1.2 至 2.0）：控制詞頻（TF）的影響，數值越大，詞頻的影響越大。
                - **b**（範圍 0 至 1，預設為 0.75）：控制文件長度的影響，當 b = 1 時，長度歸一化影響最大，當 b = 0 時，文件長度不影響計算結果。
@Note    : https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
"""


from typing import (Any,
                    Dict,
                    List,
                    Optional,
                    )
from uuid import UUID

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from infra.stores.base import SearchHit, LexicalIndexStore
from infra.stores.errors import ElasticsearchActionError

from objects import (Chunk,
                     DocumentMetadata,
                     )

class ElasticsearchBM25Store(LexicalIndexStore):
    # 若要使用 BM25 查詢文件，必須在資料儲存階段就先進行紀錄
    """Elasticsearch-based store implementation

    有保留 vector 儲存的機制，如果有輸入文本的 vector，也可以使用 `search_by_vector_similarity` 來搜尋
    """

    def __init__(self,
                 host:str="localhost",
                 port:int=9200,
                 index_name:str="default_index",
                 username:Optional[str]=None,
                 password:Optional[str]=None,
                 ssl_verify:bool=False,
                 k1:float=1.20,
                 b:float=0.75,
                 ):
        """Initialize Elasticsearch store
        
        Args:
            host (str): Elasticsearch host address
            port (int): Elasticsearch port number
            index_name (str): Name of the index to use
            username (str, optional): Username for authentication
            password (str, optional): Password for authentication
            ssl_verify (bool): Whether to verify SSL certificates
        """
        # 構建連接URL
        url = f"http://{host}:{port}"

        # 設置客戶端參數
        client_kwargs = {"hosts": [url],
                         "verify_certs": ssl_verify,
                         }

        # 添加認證信息（如果提供）
        if username and password:
            client_kwargs["basic_auth"] = (username, password)

        # 初始化客戶端
        self.client = Elasticsearch(**client_kwargs)
        self._index_name = index_name

        # BM25 參數
        self.k1 = k1
        self.b = b

        # 確保索引存在
        self._create_index_if_not_exists()

    @property
    def index_name(self):
        return self._index_name

    def _create_index_if_not_exists(self) -> None:
        """Create the index if it doesn't exist with appropriate mappings
        """
        if not self.client.indices.exists(index=self.index_name):
            settings = {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "ik_max_word",  # 使用 IK 中文分詞器（最大切詞）
                        }
                    }
                },
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": self.k1,
                        "b": self.b,
                    }
                }
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",  # 保留 IK 中文分詞器
                        "similarity": "custom_bm25",  # 使用 BM25 similarity
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1536,
                    },
                    "metadata": {
                        "properties": {
                            "file_type": {"type": "keyword"},
                            "file_name": {"type": "keyword"},
                            "metadata_created_at": {"type": "date"},
                            "title": {"type": "text"},
                            "author": {"type": "keyword"},
                            "keywords": {"type": "keyword"},
                            "total_chunks": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                            "spec_info": {"type": "object"},
                            "extra": {"type": "object"}
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name,
                                       settings=settings,
                                       mappings=mappings,
                                       )

    def recreate_index(self) -> None:
        """
        重新建立索引：先刪除現有索引，再根據預設的映射與設定重新建立索引。
        """
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self._create_index_if_not_exists()

    def _chunk_to_es_doc(self,
                            chunk:Chunk,
                            ) -> Dict[str,Any]:
        """Convert Chunk object to Elasticsearch document format"""
        return {"content": chunk.content,
                "embedding": chunk.embedding,
                "metadata": chunk.metadata.to_dict(),
                }

    def _es_doc_to_chunk(self,
                         es_doc:Dict[str,Any],
                         doc_id:UUID,
                         ) -> Chunk:
        """Convert Elasticsearch document to Chunk object"""
        return Chunk(id=doc_id,
                     content=es_doc["_source"]["content"],
                     metadata=DocumentMetadata.from_dict(es_doc["_source"]["metadata"]),
                     embedding=es_doc["_source"].get("embedding"),
                     )

    def insert(self,
               chunk:Chunk,
               ) -> None:
        """Insert a document chunk into the store
        
        Args:
            chunk (Chunk): Document chunk to insert
            
        Raises:
            ElasticsearchActionError: If document chunk already exists
        """
        try:
            # 檢查文檔是否已存在
            existing = self.client.get(index=self.index_name,
                                       id=str(chunk.id),
                                       )
            # 若存在，則報錯
            if existing:
                raise ElasticsearchActionError(f"Insert fail: Document with ID {chunk.id} already exists in the index.")
        except NotFoundError:
            # 若文檔不存在，轉換並插入文檔
            es_doc = self._chunk_to_es_doc(chunk)
            self.client.index(index=self.index_name,
                              id=str(chunk.id),
                              document=es_doc,
                              refresh=True,
                              )

    def search_by_vector_similarity(self, 
                                    query_embedding:List[float], 
                                    top_k:int=5,
                                    metadata_filter:Optional[Dict[str,Any]]=None,
                                    ) -> List[SearchHit]:
        """Search for similar document chunks using vector similarity
        
        Args:
            query_embedding (List[float]): Query embedding vector
            top_k (int): Number of results to return
            metadata_filter (Dict[str, Any], optional): Metadata filters
            
        Returns:
            List[Document]: List of matching document chunks
        """
        # 構建向量相似度查詢
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
        
        # 如果有 metadata 過濾，添加到查詢中
        if metadata_filter:
            query = {
                "bool": {
                    "must": [script_query],
                    "filter": metadata_filter
                }
            }
        else:
            query = script_query
            
        # 執行搜索
        response = self.client.search(
            index=self.index_name,
            query=query,
            size=top_k
        )
        
        # 處理結果
        results: List[SearchHit] = []
        for hit in response["hits"]["hits"]:
            chunk = self._es_doc_to_chunk(hit, UUID(hit["_id"]))
            score = hit["_score"]
            results.append(SearchHit(chunk=chunk, score=score))

        return results

    def search(self, 
               query_text:str, 
               top_k:int=5,
               metadata_filter:Optional[Dict[str, Any]]=None,
               ) -> List[SearchHit]:
        """Search for similar documents using BM25 (text-based matching)
        
        Args:
            query_text (str): User query text
            top_k (int): Number of results to return
            metadata_filter (Dict[str, Any], optional): Metadata filters
            
        Returns:
            List[Chunk]: List of matching documents
        """
        match_query = {"match": {"content": query_text}}
        if metadata_filter:
            query = {
                "bool": {
                    "must": [match_query],
                    "filter": metadata_filter
                }
            }
        else:
            query = match_query

        response = self.client.search(index=self.index_name,
                                      query=query,
                                      size=top_k,
                                      )

        results:List[SearchHit] = []
        for hit in response["hits"]["hits"]:
            chunk = self._es_doc_to_chunk(hit, UUID(hit["_id"]))
            score = hit["_score"]
            results.append(SearchHit(chunk=chunk, score=score))
        return results

    def get(self,
            chunk_id:UUID,
            ) -> List[Chunk]:
        """Get document by ID
        
        Args:
            chunk_id (UUID): Document chunk ID
            
        Returns:
            List[Chunk]: List containing the document chunk
            
        Raises:
            ElasticsearchActionError: 若沒有匹配的結果，則拋出錯誤。
        """
        try:
            response = self.client.get(index=self.index_name,
                                       id=str(chunk_id),
                                       )
            return [self._es_doc_to_chunk(response, chunk_id)]
        except NotFoundError as err:
            raise ElasticsearchActionError(f"Get fail: Document with ID {chunk_id} does not exist in the collection.") from err

    def update(self,
               chunk:Chunk,
               ) -> None:
        """Update an existing document chunk
        
        Args:
            document (Chunk): Document chunk with updated content
            
        Raises:
            ElasticsearchActionError: If document doesn't exist
        """
        try:
            self.get(chunk.id)
        except ElasticsearchActionError:
            raise ElasticsearchActionError(f"Document with ID {chunk.id} not found")
        
        es_doc = self._chunk_to_es_doc(chunk)
        self.client.update(index=self.index_name,
                           id=str(chunk.id),
                           doc=es_doc,
                           refresh=True,
                           )

    def delete(self,
               chunk_id:UUID,
               ) -> None:
        """Delete a document chunk from the store
        
        Args:
            doc_id (UUID): ID of document chunk to delete
        """
        try:
            self.client.delete(
                index=self.index_name,
                id=str(chunk_id),
                refresh=True
            )
        except NotFoundError:
            pass  # 如果文檔不存在，靜默處理

    def upsert(self,
               chunk:Chunk,
               ) -> None:
        try:
            existence = self.client.get(index=self.index_name,
                                        id=str(chunk.id),
                                        )
            if existence:
                self.update(chunk)
        except NotFoundError as err:
            self.insert(chunk)

    def __len__(self) -> int:
        """Get the number of documents in the store
        
        Returns:
            int: Number of documents
        """
        self.client.indices.refresh(index=self.index_name)
        stats = self.client.count(index=self.index_name)
        return stats["count"]

    def clean(self,
              query:Optional[Dict[str,Any]]=None,
              ) -> None:
        """Remove documents from the store
        
        Args:
            query (Dict[str, Any], optional): Query to filter documents to delete
        """
        if query:
            self.client.delete_by_query(
                index=self.index_name,
                query=query,
                refresh=True
            )
        else:
            self.client.options(ignore_status=[404]).indices.delete(index=self.index_name)
            self._create_index_if_not_exists()
