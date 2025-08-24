# personal-rag

Personal RAG, for japanese learning, math/statistic notes and other documents.

## 啟動 MCP Server 並接到 Open WebUI

### 需求

- Python 3.10+、`pipenv`
- 已安裝 `mcpo`：`pipenv install mcpo`
- Open WebUI（具「Add Tool Server / OpenAPI」功能）

### 啟動

```bash
pipenv run mcpo --port 56485 -- env PYTHONPATH=src pipenv run python src/mcp_server/jp_learning_rag.py
```

### 驗證

```bash
curl http://localhost:56485/openapi.json | jq '.paths'
```

看到有工具相關的路徑（非空）即表示正常。

### 在 Open WebUI 設定

1. 開啟：**Settings → Tools → Add Tool Server**  
2. 類型：**OpenAPI**  
3. URL：`http://localhost:56485`  
4. 儲存後應可看到工具清單


## Project Organization


src/
├── api.py                        # FastAPI 入口（/ingest, /query）
├── config.py                     # Pydantic Settings（DB/模型/參數）
├── base.py
├── objects.py                    # Document, Chunk 等資料型別（不繼承 docling）
├── pipelines/                    # 串接各步驟的實作（同步即可）
│   ├── ingest_pipeline.py        # 讀檔 → 前處理 → 分割 → 寫入
│   └── query_pipeline.py         # 重寫/HyDE → 搜尋 → rerank → 回答/引用
├── ingestion/                    # 檔案讀取：PDF/DOCX/MD/HTML/TXT/CSV
│   ├── __init__.py
│   └── loaders.py                # FileLoader, WebLoader（可先只做 File）
├── preprocess/                   # 前處理（極簡清理）
│   ├── __init__.py
│   └── cleaners.py               # 去 boilerplate、語言偵測（可選）
├── chunking/                     # ← Docling 只活在這
│   ├── __init__.py
│   └── docling.py                # Docling 讀檔+階層切塊+contextualize 薄包裝
├── postprocess/                  # 後處理（可選）
│   ├── __init__.py
│   └── answer_filters.py         # 來源標註/去重/片段拼接規則
├── embeddings/
│   ├── __init__.py
│   ├── base.py                   # 簡單介面：embed_texts(list[str])->list[list[float]]
│   ├── openai_embedder.py        # 任一雲端
│   └── bge_embedder.py           # 本地/或留存根
├── search/
│   ├── __init__.py
│   ├── hyde.py                   # 產生 HyDE 假想查詢（可選）
│   └── engines.py                # lexical / vector / hybrid 三種查詢策略
├── rerank/
│   ├── __init__.py
│   ├── base.py                   # rerank(query, passages)->scored_passages
│   └── bge_reranker.py           # 先放一種，未來再加
├── validation/
│   ├── __init__.py
│   └── schema.py                 # 入出參數的 Pydantic 模型（API 用）
├── cache/
│   ├── __init__.py
│   └── memo.py                   # 進程內快取裝飾器（嵌入/查詢）
├── infra/                        # 只保留最小基座
│   ├── __init__.py
│   ├── db.py                     # SQLAlchemy engine + session_scope() + uow()
│   └── stores/
│       ├── __init__.py
│       ├── base.py               # Store 介面（你已經有）
│       ├── pgvector.py           # 向量＋全文＋混合查詢（SQL 端融合）
│       ├── elasticsearch.py      # 先不啟用（保留檔案即可）
│       └── errors.py             # Transient/Permanent/NotFound
└── __init__.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
