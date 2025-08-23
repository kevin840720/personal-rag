# -*- encoding: utf-8 -*-
"""
@File    :  pgvector.py
@Time    :  2025/08/13 20:03:59
@Author  :  Kevin Wang
@Desc    :  None
"""

from typing import List, Optional, Sequence
from uuid import UUID

from sqlalchemy import (create_engine,
                        event,
                        Index,
                        select,
                        Text,
                        text,
                        )
from sqlalchemy.orm import declarative_base, mapped_column, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from pgvector.sqlalchemy import Vector
from pgvector.psycopg import register_vector

from infra.stores.base import VectorIndexStore, SearchHit
from objects import Chunk, DocumentMetadata


def make_vector_chunk_class(base, schema:str):
    class VectorChunk(base):
        __table_args__ = {"schema": schema}
        __tablename__ = "vector"

        chunk_id = mapped_column(PGUUID(as_uuid=True), primary_key=True)
        content = mapped_column(Text, nullable=False)
        chunk_metadata = mapped_column(JSONB, nullable=False)
        embedding = mapped_column(Vector(1536))  # NOTE: 留意此處 dim 寫死
    return VectorChunk

class PGVectorStore(VectorIndexStore):
    def __init__(self,
                 host:str="localhost",
                 port:int=5432,
                 dbname:str="postgres",
                 schema:str="default_vector_index",
                 user:str="postgres",
                 password:str="",
                 ):
        """初始化 PGVectorStore，連接 PostgreSQL 並設定向量儲存環境。

        Args:
            host (str, optional): PostgreSQL 主機位址，預設 "localhost"
            port (int, optional): PostgreSQL 埠號，預設 5432
            dbname (str, optional): 資料庫名稱，預設 "postgres"
            user (str, optional): 資料庫使用者，預設 "postgres"
            password (str, optional): 資料庫密碼，預設空字串
        """
        self._Base = declarative_base()
        self.VectorChunk = make_vector_chunk_class(self._Base, schema)
        dsn = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(dsn, echo=False)

        # 1. 因為 PostgreSQL 本身沒有 vector 型態，所以要先在 Psycopg3 註冊
        #    event.listens_for(self.engine, "connect") 會在 Engine 建立新的資料庫連線時觸發（此處的 'connect' 不是 method name）
        #    並呼叫 connect() 把 pgvector 的型態 (vector, halfvec, bit, sparsevec) 註冊到 Psycopg3。
        @event.listens_for(self.engine, "connect")
        def register_pgvector_type(dbapi_connection, connection_record):
            register_vector(dbapi_connection)

        # 2. 資料庫環境設定
        # 2.1 建立 schema
        with self.engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        # 2.2 啟用 vector extension (NOTE: vector extension 已經在 .database/postgres/init/01-create-vector.sql 中啟用，這句可以考慮移除)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # 3. 將已經註冊在 Base 中的結構，在資料庫中建表
        self._Base.metadata.create_all(self.engine)

        # 4. 建立向量索引（一個幫你加速搜尋的 index 結構）
        #    HNSW 建立慢又佔空間，但搜尋快
        index = Index("vector_index_hnsw",
                      self.VectorChunk.embedding,
                      postgresql_using="hnsw",
                      postgresql_with={"m": 16, "ef_construction": 64},
                      postgresql_ops={"embedding": "vector_cosine_ops"},
                      )
        index.create(self.engine, checkfirst=True)

        # 5. 建立 Session 工廠
        self.Session = sessionmaker(bind=self.engine)

    def insert(self, chunk: Chunk) -> None:
        with self.Session() as session:
            session.add(self.VectorChunk(chunk_id=chunk.id,
                                         content=chunk.content,                   # str
                                         chunk_metadata=chunk.metadata.to_dict(), # JSONB
                                         embedding=chunk.embedding,               # Vector
                                         ))
            session.commit()

    def search(self,
               query_embedding:Sequence[float],
               top_k:int=5,
               ) -> List[SearchHit]:
        with self.Session() as session:
            stmt = (select(self.VectorChunk,
                           self.VectorChunk.embedding.l2_distance(query_embedding).label("score")
                           )
                    .order_by("score")  # 距離小的在前面
                    .limit(top_k)
                    )
            rows = session.execute(stmt).all()

        return [SearchHit(chunk=Chunk(id=row.VectorChunk.chunk_id,
                                      content=row.VectorChunk.content,
                                      metadata=DocumentMetadata.from_dict(row.VectorChunk.chunk_metadata),  # 你要有 from_dict 方法
                                      embedding=row.VectorChunk.embedding,
                                      ),
                          score=row.score,
                          )
                for row in rows
                ]

    def get(self, chunk_id:UUID) -> Optional[Chunk]:
        with self.Session() as session:
            vc = session.get(self.VectorChunk, chunk_id)
            if not vc:
                return None
            return Chunk(id=vc.chunk_id,
                         content=vc.content,
                         metadata=DocumentMetadata.from_dict(vc.chunk_metadata),
                         embedding=vc.embedding,
                         )

    def update(self, chunk:Chunk) -> None:
        with self.Session() as session:
            vc = session.get(self.VectorChunk, chunk.id)
            if not vc:
                raise ValueError(f"Chunk {chunk.id} not found.")

            vc.content = chunk.content                       # str
            vc.chunk_metadata = chunk.metadata.to_dict()     # dict -> JSONB
            vc.embedding = chunk.embedding                   # list[float] -> Vector

            session.commit()
        
        # NOTE: 缺少 update 失敗的處理機制？

    def delete(self, chunk_id:UUID) -> None:
        with self.Session() as session:
            vc = session.get(self.VectorChunk, chunk_id)
            if vc:
                session.delete(vc)
                session.commit()
