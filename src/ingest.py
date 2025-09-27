# -*- encoding: utf-8 -*-
"""
@File    :  ingest.py
@Time    :  2025/09/20 20:34:16
@Author  :  Kevin Wang
@Desc    :  文件 ingestion pipeline：
            1. 以 checksum/metadata 登記或更新 catalog
            2. 建立 IngestTask 鎖定 document/task ID
            3. loader 解析檔案並交由 chunker 切塊
            4. 對每個 chunk 執行 OpenAI 嵌入並 upsert PGVector
            5. 同步 chunk 內容到 Elasticsearch BM25
            6. 任務完成回寫統計；任何階段失敗就標記 failed 後回拋例外
@Example :  pipenv run python src/ingest.py --path ./data/raw/sample.pdf --pg-schema sample
"""


from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
import uuid

from dotenv import load_dotenv
from logging_.logger import get_logger
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from chunking.docling import DoclingChunkProcessor
from chunking.no_chunk import NoChunkProcessor
from embedding.openai_embed import OpenAIEmbeddingModel
from infra.stores.elasticsearch import ElasticsearchBM25Store
from infra.stores.pgvector import PGVectorStore
from ingestion.base import LoaderResult
from ingestion.file_loaders.markdown import DoclingMarkdownLoader
from ingestion.file_loaders.pdf import DoclingPDFLoader
from ingestion.file_loaders.goodnotes import GoodnotesLoader
from objects import Chunk

load_dotenv()

LOGGER = get_logger(__name__)

CATALOG_SCHEMA = "ingest_catalog"

# --------------------------------------------------------------------------------------
# Catalog ORM definitions
# --------------------------------------------------------------------------------------


class CatalogBase(DeclarativeBase):
    pass


class DocumentRecord(CatalogBase):
    __tablename__ = "documents"
    __table_args__ = {"schema": CATALOG_SCHEMA}

    document_id:Mapped[uuid.UUID]=mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_path:Mapped[str]=mapped_column(Text, unique=True, nullable=False)
    file_name:Mapped[str]=mapped_column(Text, nullable=False)
    checksum:Mapped[str]=mapped_column(String(64), nullable=False)
    version_tag:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    filesize_bytes:Mapped[int]=mapped_column(Integer, nullable=False, default=0)
    mime_type:Mapped[Optional[str]]=mapped_column(String(255), nullable=True)
    metadata_json:Mapped[Optional[dict]]=mapped_column(JSONB, nullable=True)
    status:Mapped[str]=mapped_column(String(32), nullable=False, default="pending")
    needs_reingest:Mapped[bool]=mapped_column(Boolean, nullable=False, default=True)
    last_ingested_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    vector_synced_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    lexical_synced_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    chunk_count:Mapped[int]=mapped_column(Integer, nullable=False, default=0)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    tasks:Mapped[List["IngestTaskRecord"]]=relationship(back_populates="document", cascade="all, delete-orphan")


class IngestTaskRecord(CatalogBase):
    __tablename__ = "tasks"
    __table_args__ = {"schema": CATALOG_SCHEMA}

    task_id:Mapped[uuid.UUID]=mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id:Mapped[str]=mapped_column(PGUUID(as_uuid=True), ForeignKey(f"{CATALOG_SCHEMA}.documents.document_id", ondelete="CASCADE"), nullable=False)
    trigger_type:Mapped[str]=mapped_column(String(32), nullable=False)
    status:Mapped[str]=mapped_column(String(32), nullable=False, default="pending")
    retry_count:Mapped[int]=mapped_column(Integer, nullable=False, default=0)
    started_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    finished_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    error_message:Mapped[Optional[str]]=mapped_column(Text)
    chunk_count:Mapped[int]=mapped_column(Integer, nullable=False, default=0)
    vector_sync_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    lexical_sync_at:Mapped[Optional[datetime]]=mapped_column(DateTime(timezone=True))
    log_ref:Mapped[Optional[str]]=mapped_column(String(128))
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    document:Mapped[DocumentRecord]=relationship(back_populates="tasks")


# --------------------------------------------------------------------------------------
# Helper dataclasses and pure helpers
#
#   1. 啟動前置階段（環境與 catalog 準備）
#       - build_pg_engine、ensure_schema 在模組載入時就跑，確保 catalog schema 準備好。
#       - build_embedder、build_vector_store、build_lexical_store 在實際執行 ingest 前建立好外部資源。
#   2. 任務開始（catalog 登記）
#       - compute_checksum、detect_mime 在進入 pipeline 時算出檔案指紋與 MIME。
#       - register_document 用這些資訊更新/新增 DocumentRecord。這邊會判斷是否需要重跑。
#       - start_task 緊接著建立 IngestTaskRecord，標記狀態為 running。
#   3. 內容處理（載入 → chunk → embedding → 索引）
#       - choose_loader 依 --file-type 或副檔名挑對 loader（目前支援 pdf/md/goodnotes）。
#       - choose_chunk_processor 根據 loader 輸出決定 chunker（Docling 或 NoChunk）。
#       - 進入迴圈後，每一個 chunk 先跑 embedder.encode，接著呼叫 upsert_vector_store 與 lexical_store.upsert 送進兩
#   個索引。
#   4. 成功收尾
#       - 所有 chunk 完成後，mark_task_success 更新任務狀態與 chunk 數、同步時間。
#       - update_document_after_success 在文件層標記 ingested 並填入最後同步時間。
#       - 兩者完成才 session.commit() 結束本次任務。
#   5. 失敗回滾
#       - 只要在處理過程拋例外，會落到 except 區塊：
#           - mark_task_failure 把任務狀態設為 failed 並記錄錯誤訊息。
#           - mark_document_failure 對應的 DocumentRecord 也被標記為 failed、needs_reingest=True。
#           - 這一段最後一樣 session.commit()，讓 catalog 詳實記下失敗紀錄。
# --------------------------------------------------------------------------------------


class IngestInputs(BaseModel):
    file_path:Path
    checksum:str
    filesize:int
    mime_type:Optional[str]


def build_pg_engine():
    """建立與主資料庫的 SQLAlchemy Engine。

    Returns:
        Engine: 用於 catalog 操作的 SQLAlchemy Engine 物件。
    """
    host = os.getenv("MY_POSTGRE_HOST", "localhost")
    port = os.getenv("MY_POSTGRE_PORT", "5432")
    dbname = os.getenv("MY_POSTGRE_DB_NAME", "postgres")
    user = os.getenv("MY_POSTGRE_USERNAME", "postgres")
    password = os.getenv("MY_POSTGRE_PASSWORD", "")
    dsn = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(dsn, echo=False, pool_pre_ping=True)


def ensure_schema(engine) -> None:
    """確保 `ingest_catalog` schema 存在並同步 ORM 結構。

    Args:
        engine (Engine): SQLAlchemy Engine 實例。
    """
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{CATALOG_SCHEMA}"'))
    CatalogBase.metadata.create_all(engine)


ENGINE = build_pg_engine()
SessionFactory = sessionmaker(bind=ENGINE)
ensure_schema(ENGINE)


def compute_checksum(path:Path,
                     block_size:int=1024*1024,
                     ) -> str:
    """計算檔案內容的 SHA-256 雜湊值。

    Args:
        path (Path): 目標檔案路徑。
        block_size (int): 讀檔區塊大小 (位元組)。

    Returns:
        str: 檔案內容的十六進位 checksum。
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_mime(path:Path) -> Optional[str]:
    """推斷檔案的 MIME 類型。

    Args:
        path (Path): 目標檔案路徑。

    Returns:
        Optional[str]: 推斷出的 MIME 類型，失敗時為 None。
    """
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def register_document(session,
                      inputs:IngestInputs,
                      ) -> DocumentRecord:
    """在 catalog 中建立或更新文件紀錄。

    Args:
        session (Session): SQLAlchemy Session 物件。
        inputs (IngestInputs): 檔案基本資訊。

    Returns:
        DocumentRecord: 目前最新的文件紀錄。
    """
    record = session.query(DocumentRecord).filter(DocumentRecord.source_path == str(inputs.file_path)).one_or_none()
    now = datetime.now(timezone.utc)
    if record is None:
        record = DocumentRecord(source_path=str(inputs.file_path),
                                 file_name=inputs.file_path.name,
                                 checksum=inputs.checksum,
                                 filesize_bytes=inputs.filesize,
                                 mime_type=inputs.mime_type,
                                 status="pending",
                                 needs_reingest=True,
                                 )
        session.add(record)
        session.flush()
        record.created_at = now
        record.updated_at = now
        return record

    if record.checksum != inputs.checksum:
        record.checksum = inputs.checksum
        record.filesize_bytes = inputs.filesize
        record.mime_type = inputs.mime_type
        record.status = "pending"
        record.needs_reingest = True
        record.updated_at = now
    return record


def start_task(session,
               document:DocumentRecord,
               trigger_type:str,
               ) -> IngestTaskRecord:
    """在 catalog 中建立新的 ingest 任務紀錄。

    Args:
        document (DocumentRecord): 目標文件紀錄。
        trigger_type (str): 觸發來源描述。

    Returns:
        IngestTaskRecord: 新建立的任務紀錄。
    """
    task = IngestTaskRecord(document=document,
                            trigger_type=trigger_type,
                            status="running",
                            started_at=datetime.now(timezone.utc),
                            )
    session.add(task)
    session.flush()
    return task


def mark_task_success(task:IngestTaskRecord,
                      *,
                      chunk_count:int,
                      vector_ts:datetime,
                      lexical_ts:datetime,
                      ) -> None:
    """將任務標記為成功並更新統計資訊。

    Args:
        task (IngestTaskRecord): 需要更新的任務紀錄。
        chunk_count (int): 本次 ingest 產生的 chunk 數量。
        vector_ts (datetime): 向量索引同步時間。
        lexical_ts (datetime): 關鍵字索引同步時間。
    """
    task.status = "success"
    task.chunk_count = chunk_count
    task.finished_at = datetime.now(timezone.utc)
    task.vector_sync_at = vector_ts
    task.lexical_sync_at = lexical_ts


def mark_task_failure(task:IngestTaskRecord,
                      error:Exception,
                      ) -> None:
    """將任務標記為失敗並記錄錯誤訊息。

    Args:
        task (IngestTaskRecord): 需要更新的任務紀錄。
        error (Exception): 捕捉到的例外。
    """
    task.status = "failed"
    task.error_message = str(error)
    task.finished_at = datetime.now(timezone.utc)
    task.retry_count += 1


def update_document_after_success(document:DocumentRecord,
                                  *,
                                  chunk_count:int,
                                  vector_ts:datetime,
                                  lexical_ts:datetime,
                                  ) -> None:
    """在任務成功後更新文件層級的統計資訊。

    Args:
        document (DocumentRecord): 目標文件紀錄。
        chunk_count (int): 本次 ingest 產出的 chunk 數量。
        vector_ts (datetime): 向量索引同步時間。
        lexical_ts (datetime): 關鍵字索引同步時間。
    """
    document.status = "ingested"
    document.needs_reingest = False
    document.chunk_count = chunk_count
    document.last_ingested_at = datetime.now(timezone.utc)
    document.vector_synced_at = vector_ts
    document.lexical_synced_at = lexical_ts


def mark_document_failure(document:DocumentRecord,
                          ) -> None:
    """在任務失敗時標記文件狀態並要求重跑。

    Args:
        document (DocumentRecord): 目標文件紀錄。
    """
    document.status = "failed"
    document.needs_reingest = True


def choose_loader(path:Path,
                  file_type:Optional[str]=None,
                  ):
    """依檔案副檔名或使用者指定類型挑選 loader。

    Args:
        path (Path): 來源檔案路徑。
        file_type (Optional[str]): 指定 loader 類型，若為 None 則依副檔名推斷。

    Returns:
        DocumentLoader: 可處理該檔案的 Loader 實例。

    Raises:
        ValueError: 當類型或副檔名不受支援時。
    """
    kind = (file_type or path.suffix.lstrip('.')).lower()
    if kind in ("md", "markdown"):
        return DoclingMarkdownLoader()
    if kind == "pdf":
        return DoclingPDFLoader()
    if kind in ("goodnotes", "gn"):
        from ingestion.file_loaders.goodnotes.loader import GoodnotesLoader
        return GoodnotesLoader()
    if file_type is None:
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    raise ValueError(f"Unsupported file type: {file_type}")


def choose_chunk_processor(result:LoaderResult):
    """根據 loader 輸出挑選合適的 chunk 處理器。

    Args:
        result (LoaderResult): LoaderResult 物件。

    Returns:
        ChunkProcessor: 與文件格式相符的 chunk 處理器。
    """
    if result.doc and result.doc.__class__.__name__.lower().startswith("docling"):
        return DoclingChunkProcessor()
    return NoChunkProcessor()


def build_embedder() -> OpenAIEmbeddingModel:
    """建立 OpenAI 向量模型實例。

    Returns:
        OpenAIEmbeddingModel: 封裝後的向量模型。

    Raises:
        RuntimeError: 當環境變數缺少 API key。
    """
    api_key = os.getenv("OPEN_AI_API") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPEN_AI_API / OPENAI_API_KEY")
    return OpenAIEmbeddingModel(api_key=api_key)


def build_vector_store(schema:str) -> PGVectorStore:
    """建立 PGVectorStore 連線。

    Args:
        schema (str): 目標 PostgreSQL schema 名稱。

    Returns:
        PGVectorStore: 封裝好的向量索引存取器。
    """
    return PGVectorStore(host=os.getenv("MY_POSTGRE_HOST", "localhost"),
                         port=int(os.getenv("MY_POSTGRE_PORT", "5432")),
                         dbname=os.getenv("MY_POSTGRE_DB_NAME", "postgres"),
                         schema=schema,
                         user=os.getenv("MY_POSTGRE_USERNAME", "postgres"),
                         password=os.getenv("MY_POSTGRE_PASSWORD", ""),
                         )


def build_lexical_store(index_name:str) -> ElasticsearchBM25Store:
    """建立 Elasticsearch BM25 存取器。

    Args:
        index_name (str): 目標索引名稱。

    Returns:
        ElasticsearchBM25Store: 封裝好的關鍵字索引存取器。
    """
    resolved_index = index_name or os.getenv("MY_ELASTIC_INDEX", "japanese-learning")
    return ElasticsearchBM25Store(host=os.getenv("MY_ELASTIC_HOST", "localhost"),
                                  port=int(os.getenv("MY_ELASTIC_PORT", "9200")),
                                  index_name=resolved_index,
                                  username=os.getenv("MY_ELASTIC_USERNAME"),
                                  password=os.getenv("MY_ELASTIC_PASSWORD"),
                                  )


def upsert_vector_store(store:PGVectorStore,
                      chunk:Chunk,
                      ) -> None:
    """將 chunk 寫入或更新至向量索引。

    Args:
        store (PGVectorStore): PGVectorStore 實例。
        chunk (Chunk): 準備入庫的 Chunk。
    """
    existing = store.get(chunk.id)
    if existing:
        store.update(chunk)
    else:
        store.insert(chunk)


def ingest_single_file(file_path:Path,
                      *,
                      trigger_type:str="manual_cli",
                      pg_schema:str,
                      es_index:Optional[str]=None,
                      file_type:Optional[str]=None,
                      force_reingest:bool=False,
                      ) -> None:
    """執行單一檔案的 ingest 流程。

    Args:
        file_path (Path): 來源檔案路徑。
        trigger_type (str): 記錄在 catalog 的任務觸發來源。
        pg_schema (str, optional): 向量索引所屬 PostgreSQL schema。
        es_index (Optional[str], optional): 關鍵字索引名稱，省略時為 schema 的小寫。
        file_type (Optional[str], optional): 手動指定 loader 類型。
        force_reingest (bool, optional): 強制忽略 catalog 狀態重新 ingest。

    Raises:
        FileNotFoundError: 當檔案不存在時。
        Exception: 流程中出現未處理例外時。
    """
    # 確認來源檔案存在
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # 準備檔案 checksum 與基本 metadata
    checksum = compute_checksum(file_path)
    filesize = file_path.stat().st_size
    mime_type = detect_mime(file_path)
    inputs = IngestInputs(file_path=file_path,
                          checksum=checksum,
                          filesize=filesize,
                          mime_type=mime_type,
                          )

    # 在 catalog 登記文件並檢查是否需要重新執行
    with SessionFactory() as session:
        document = register_document(session, inputs)
        if force_reingest:
            LOGGER.info("Force reingest enabled, overriding catalog state: %s", file_path)
            document.needs_reingest = True
            document.status = "pending"

        if not document.needs_reingest and document.status == "ingested":
            LOGGER.info("Document up-to-date, skipping: %s", file_path)
            return

        task = start_task(session, document, trigger_type)
        task_id = task.task_id
        document_id = document.document_id
        session.commit()

    # 建立嵌入模型與索引存取器
    embedder = build_embedder()
    resolved_schema = pg_schema
    vector_store = build_vector_store(pg_schema)
    resolved_index = es_index or resolved_schema.lower()
    lexical_store = build_lexical_store(resolved_index)

    # 依檔案類型挑 loader，並展開主要處理流程
    try:
        loader = choose_loader(file_path, file_type=file_type)
        loader_results = loader.load(str(file_path))

        total_chunks = 0
        vector_ts = datetime.now(timezone.utc)
        lexical_ts = datetime.now(timezone.utc)

        for result in loader_results:
            chunker = choose_chunk_processor(result)
            if isinstance(chunker, DoclingChunkProcessor):
                if result.doc is None:
                    raise ValueError('Docling chunker requires doc object')
                chunks = chunker.process(result.doc, metadata=result.metadata)
            else:
                chunks = chunker.process(result.doc or result.content,
                                         metadata=result.metadata,
                                         content=result.content,
                                         )
            for chunk in chunks:
                chunk.embedding = embedder.encode(chunk.content)
                upsert_vector_store(vector_store, chunk)
                lexical_store.upsert(chunk)
            total_chunks += len(chunks)

        vector_ts = datetime.now(timezone.utc)
        lexical_ts = vector_ts

    except Exception as exc:
        # 失敗就回滾
        with SessionFactory() as session:
            document = session.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).one()
            task = session.query(IngestTaskRecord).filter(IngestTaskRecord.task_id == task_id).one()
            mark_task_failure(task, exc)
            mark_document_failure(document)
            session.commit()
        raise

    # 任務成功時回寫任務與文件狀態
    with SessionFactory() as session:
        document = session.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).one()
        task = session.query(IngestTaskRecord).filter(IngestTaskRecord.task_id == task_id).one()
        mark_task_success(task, chunk_count=total_chunks, vector_ts=vector_ts, lexical_ts=lexical_ts)
        update_document_after_success(document, chunk_count=total_chunks, vector_ts=vector_ts, lexical_ts=lexical_ts)
        session.commit()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv:Optional[Iterable[str]]=None):
    """解析 CLI 參數。

    Args:
        argv (Optional[Iterable[str]]): 可選的參數列表。

    Returns:
        Namespace: argparse 解析結果。
    """
    parser = argparse.ArgumentParser(description="Document ingest pipeline")
    parser.add_argument("--path", required=True, help="Path to the source file")
    parser.add_argument("--trigger", default="manual_cli", help="Trigger type for catalog logging")
    parser.add_argument("--pg-schema", required=True, help="PostgreSQL schema for vector store catalog linkage")
    parser.add_argument("--es-index", default=None, help="Elasticsearch index name (defaults to lower-cased pg schema)")
    parser.add_argument("--file-type", default=None, help="Explicit loader selection, e.g. pdf/md/goodnotes")
    parser.add_argument("--force-reingest", action="store_true", help="Force ingestion even if catalog marks document as ingested")
    return parser.parse_args(argv)


def main(argv:Optional[Iterable[str]]=None) -> int:
    """CLI 入口函式。

    Args:
        argv (Optional[Iterable[str]]): 可選的參數列表。

    Returns:
        int: 程式結束代碼。
    """
    arg_list = list(argv) if argv is not None else sys.argv[1:]
    if any(arg in ("-h", "--help") for arg in arg_list):
        try:
            parse_args(arg_list)
        except SystemExit as exc:
            return exc.code or 0
        return 0

    args = parse_args(arg_list)
    file_path = Path(args.path).expanduser().resolve()
    try:
        ingest_single_file(file_path,
                           trigger_type=args.trigger,
                           pg_schema=args.pg_schema,
                           es_index=args.es_index,
                           file_type=args.file_type,
                           force_reingest=args.force_reingest,
                           )
    except Exception as exc:
        LOGGER.error("Ingestion failed: %s", exc, exc_info=True)
        return 1
    LOGGER.info("Ingestion completed: %s", file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
