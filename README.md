# personal-rag

面向「私人筆記（目前只處理日文筆記）」的檢索增強生成（RAG）系統。

本專案包含：

- MCP 工具伺服器：提供檢索工具（向量 + 關鍵字 BM25）給外部 Agent 使用
- 以 OpenAI 相容介面對接 OpenWebUI
- GoodNotes PDF/圖片 OCR 與筆記抽取管線：將手寫筆記與掃描教材轉為可檢索文本

整體串接：（OpenAPI 工具）↔ OpenWebUI ↔ MCP Server

## 初始化指南

## TL;DR

```bash
# 設定專案參數
cp .env.example .env
# 專案環境設定：Python
pipenv install --python 3.11
pipenv install --dev
# 專案環境設定：資料庫
chmod +x docker-start-up.sh
./docker-start-up.sh
# 資料載入
pipenv run env PYTHONPATH=src python src/ingest.py --path ./data/raw/sample.pdf --pg-schema Japanese-Learning
# 啟動服務(以下則一)
pipenv run env PYTHONPATH=src python src/app.py
pipenv run mcpo --port 56485 -- env PYTHONPATH=src pipenv run python src/mcp_server/jp_learning_rag.py
```

### 必備環境

- Python 3.11 與 `pipenv`
- Docker 與 Docker Compose（啟動 PostgreSQL、Redis、Elasticsearch、Kibana）
- `curl`、`jq`
- GoodNotes/PDF OCR 需求：系統需安裝 `poppler`（匯出 `pdf2image`），另電腦需安裝對應 CUDA driver。

### 安裝依賴

```bash
pipenv install --python 3.11
pipenv install --dev
```

> 提醒：專案所有指令預設透過 `pipenv run` 執行，部分需要另外設定 `PYTHONPATH=src`。

### 設定環境變數

1. 複製範本：`cp .env.example .env`
2. 編輯 `.env` 並填入必要值。

關鍵欄位摘要：

| 變數 | 說明 |
| ---- | ---- |
| `OPEN_AI_API` | OpenAI API Key，`text-embedding-3-small` 會使用。 |
| `VOLUME_PATH` | Docker Volume 目錄，預設 `./.database`。腳本會複製初始化 SQL 與整理權限。 |
| `MY_REDIS_*` | Redis 主機、埠、密碼（快取嵌入向量）。 |
| `MY_POSTGRE_*` | PostgreSQL + pgvector 連線資訊。`JapaneseLearningRAGServer` 預設 schema 為 `Japanese-Learning`。 |
| `MY_ELASTIC_*` | Elasticsearch 認證與位置。首次啟動後會安裝 IK Analyzer 插件。 |
| `MY_KIBANA_PORT` | 如需 Kibana UI，保留預設即可。 |
| `LANGFUSE_*` | （選填）啟用 Langfuse 監控。缺值時相關功能自動停用。 |

### 啟動基礎服務

專案提供 `docker-start-up.sh` 封裝初始化與插件設定流程。

```bash
chmod +x docker-start-up.sh
./docker-start-up.sh            # 首次啟動或需要安裝/更新 IK 插件
# ./docker-start-up.sh --recreate  # 需要清空 Elasticsearch 資料夾時使用
```

#### 備註：腳本內容說明

- 讀取 `.env`，複製 `./.database/postgres/init/01-create-vector.sql` 至 volume，確保 pgvector extension 可用。
- 整理 Elasticsearch 資料夾權限，再執行 `docker compose up -d`。
- 等待初始化後自動安裝 `analysis-ik` 插件並重新啟動 Elasticsearch。

驗證方式：

- `docker ps` 檢查四個容器是否都在 `Up` 狀態。
- `curl http://localhost:${MY_ELASTIC_PORT}/_cluster/health | jq` 確認 Elasticsearch 正常。
- `psql postgresql://user:pass@localhost:${MY_POSTGRE_PORT}/${MY_POSTGRE_DB_NAME}` 查詢 schema 是否存在（`\dn` 應看到 `Japanese-Learning` 與 `ingest_catalog`）。

### 匯入資料到索引 (Ingestion Pipeline)

1. 將原始檔案放到 `.env` 中 `VOLUME_PATH` 之外的資料夾（例如 `data/raw/`）。
2. 以 CLI 執行 `src/ingest.py`：

```bash
pipenv run env PYTHONPATH=src python src/ingest.py \
  --path ./data/raw/sample.pdf \
  --pg-schema "Japanese-Learning" \
  --file-type goodnotes            # 可省略，會依副檔名自動判斷

# 常用額外參數：
#   --es-index japanese-learning   # 預設為 schema.lower()
#   --force-reingest               # 無視 catalog 狀態重新匯入
```

#### 備註：Ingestion Pipeline 行為說明

- 在 PostgreSQL `ingest_catalog` schema 建立/更新文件紀錄。
- 使用對應的 loader + chunker（GoodNotes 管線請參考 `src/ingestion/file_loaders/goodnotes/readme.md` 或 `notebooks/goodnotes.ipynb`）。
- 產生向量後寫入 `Japanese-Learning.vector` 表，並同步文本到 Elasticsearch `japanese-learning` index。

### 啟動 MCP 伺服器

兩種傳輸模式可依需求選用：

#### Streamable HTTP（原生 MCP 傳輸）

```bash
pipenv run env PYTHONPATH=src python src/app.py
```

- 預設監聽 `http://localhost:56481/mcp`
- 若需調整 host/port/path，編輯 `src/app.py` 中 `JapaneseLearningRAGServer(host=..., port=...)` 與 `streamable_http_path`。

#### mcpo（基於 stdio，配合 mcpo / OpenAPI 工具）

```bash
pipenv run mcpo --port 56485 -- env PYTHONPATH=src pipenv run python src/mcp_server/jp_learning_rag.py
```

- `mcpo` 會將 MCP stdio 轉成 OpenAPI Tool Server，供 OpenWebUI 等客戶端掛載。
- `--port` 為 mcpo 對外 HTTP 服務埠（預設 56485）。

### 驗證服務

確定服務是否正在運行

#### Streamable HTTP

```bash
# 建立連線，獲取 Session ID
$ curl -i -sS -N http://localhost:56481/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc":"2.0",
    "id":"1",
    "method":"initialize",
    "params":{
      "protocolVersion":"2025-06-18",
      "clientInfo":{"name":"curl","version":"8"},
      "capabilities":{"sampling":{}, "roots":{"listChanged":true}}
    }
  }'
```

成功會得到 `result.session` (此處以 `$SESSION_ID` 標註)；後續請攜帶 `Mcp-Session-Id` 呼叫 `tools/list` 與 `tools/call`。

```bash
# 完成初始化
curl -sS -X POST http://localhost:56481/mcp \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# 列工具
curl -sS -X POST http://localhost:56481/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H 'Mcp-Session-Id: $SESSION_ID' \
  -d '{"jsonrpc":"2.0","id":"2","method":"tools/list"}'

# 呼叫 search_japanese_note
curl -sS -X POST http://localhost:56481/mcp \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H 'Mcp-Session-Id: $SESSION_ID' \
  -d '{
    "jsonrpc":"2.0",
    "id":"3",
    "method":"tools/call",
    "params":{
      "name":"search_japanese_note",
      "arguments":{
        "query":"奧運是什麼？",
        "keywords":["奧運","オリンピック"],
        "top_embedding_k":3,
        "top_keyword_k":3
      }
    }
  }'
```

#### OpenAPI / mcpo

```bash
curl http://localhost:56485/openapi.json | jq '.paths'
```

若能看到 `tools` 與 `call` 等路徑即代表 MCP 工具已註冊。

## 功能特色

- 向量檢索 + 關鍵字檢索：
  - 向量：PostgreSQL + pgvector（`text-embedding-3-small`，附 Redis 嵌入快取）
  - 關鍵字：Elasticsearch BM25
- MCP 工具：`search_japanese_note` 同時支援語意（query）與關鍵字（keywords）檢索，並合併去重
- GoodNotes 文件處理：黑底手寫與白底掃描皆支援，含自動前處理、偵測、辨識、分群與主幹文字萃取
- OpenWebUI 整合：
  - 以 OpenAPI「工具伺服器」引入 MCP 工具
- 評估支持：提供 RAGAS 檢索評估腳本，便於觀察檢索純度與覆蓋率

## Appendix

## 與 OpenWebUI 整合(限 mcpo 模式)

1. 啟動 mcpo 命令後，開啟 OpenWebUI → Settings → Tools。
2. 點選 **Add Tool Server**，類型選 **OpenAPI**。

## Langfuse 監控（可選）

- 設定 `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` 後，`monitoring.langfuse_client.observe_if_enabled` 會自動包裝檢索流程。
- 可登入 Langfuse 觀察 `mcp.JapaneseLearningRAGServer.retrieve_chunks` 與 `search_japanese_note` 的輸入輸出與執行耗時。

### GoodNotes 支援說明

- 針對 GoodNotes 的 PDF/圖片，提供完整 OCR 與主幹文字萃取管線：
  - 黑底手寫與白底掃描皆支援，多種前處理（反白、去色、加強灰階、顏色過濾等）
  - 偵測（DET）→ 切塊 → 辨識（REC）→ 鄰近群組合併 → 主幹文字輸出
  - 取得每頁 metadata（頁碼、書籤 outlines 等）
- 快速上手與範例程式請見：`src/ingestion/file_loaders/goodnotes/readme.md`

## Project Organization

- Ingestion 與 catalog：`src/ingest.py`、`src/ingestion/` 內含所有 CLI 流程與 GoodNotes 管線，負責完成資料壓片、寫入 PostgreSQL 與 Elasticsearch。
- MCP 伺服器入口：`src/app.py`、`src/mcp_server/` 封裝 RAG 查詢邏輯與對外 API，提供 streamable-http 與 mcpo 兩種啟動模式。
- 評估工具：`src/evaluation/` 提供回答相關性、扎實度等評測腳本，搭配 notebooks/ 追蹤品質。
- 監控與日誌：`src/monitoring/`、`src/logging_/` 紀錄 Langfuse 指標與統一 logger 設定，便於調校與排錯。
- Notebook 範例：`notebooks/` 收集手動追蹤流程與資料前處理範例，輔助分析與除錯。

```text
personal-rag/
├── Pipfile                         # pipenv 依賴設定
├── Pipfile.lock                    # 依賴版本鎖定
├── README.md                       # 專案說明
├── docker-compose.yaml             # Docker 服務編排
├── docker-start-up.sh              # 基礎服務啟動腳本
├── notebooks/                      # Jupyter 筆記本
│   ├── example.ipynb
│   ├── file_loader.ipynb
│   └── goodnotes.ipynb             # Goodnotes 筆記 Ingest Pipeline 逐步拆解流程
└── src/                            # 主要程式碼
    ├── app.py                      # MCP streamable-http 入口
    ├── cache/                      # 快取層抽象
    │   ├── base.py                 # 快取介面定義
    │   ├── errors.py               # 快取相關例外
    │   └── redis.py                # Redis 快取實作
    ├── chunking/                   # 切段策略
    │   ├── base.py                 # 切段介面
    │   ├── docling.py              # Docling 切段流程
    │   └── no_chunk.py             # 無切段備用策略
    ├── embedding/                  # 向量嵌入
    │   ├── base.py                 # 嵌入介面
    │   └── openai_embed.py         # OpenAI Embedding 客戶端
    ├── evaluation/                 # 評估流程
    │   ├── answer_relevance.py     # 回答相關性
    │   ├── context_relevance.py    # 上下文對齊度
    │   ├── groundedness.py         # 事實扎實度檢查
    │   └── utils.py                # 評估工具函式
    ├── infra/                      # 資料儲存抽象
    │   ├── base.py                 # 儲存介面定義
    │   └── stores/
    │       ├── base.py             # 儲存層統一行為
    │       ├── elasticsearch.py    # Elasticsearch 客戶端
    │       ├── errors.py           # 儲存相關例外
    │       └── pgvector.py         # PostgreSQL/pgvector 存取
    ├── ingestion/                  # 檔案載入與 metadata 管線
    │   ├── base.py                 # Ingestion 作業骨架
    │   ├── file_loaders/
    │   │   ├── goodnotes.py        # GoodNotes 筆記解析
    │   │   ├── image.py            # 一般圖片 OCR
    │   │   ├── markdown.py         # Markdown 文件載入
    │   │   ├── office_docx.py      # Word 文件載入
    │   │   ├── office_excel.py     # Excel 表單載入
    │   │   └── pdf.py              # PDF 文件載入
    │   ├── file_loaders/goodnotes/
    │   │   ├── loader.py           # GoodNotes 專用 loader 入口
    │   │   ├── ops.py              # 前處理與切塊操作
    │   │   └── pipeline.py         # 完整 GoodNotes Pipeline
    │   └── utils.py                # Ingestion 輔助函式
    ├── ingest.py                   # Ingestion Pipeline CLI 入口
    ├── logging_/
    │   └── logger.py               # Logger 設定
    ├── mcp_server/                 # MCP 伺服器實作
    │   └── jp_learning_rag.py      # 日文筆記 RAG Server
    ├── monitoring/                 # 遙測與追蹤
    │   └── langfuse_client.py      # Langfuse 客戶端
    └── objects.py                  # 核心資料型別
```
<!-- 
## 備註

### OCR 模型超量佔用 GPU 空間

TL;DR: **如果你頻繁 OOM，再用 CPU Inference，不要用 GPU**，實測 GPU 對 20 image 預測約 35s (含載入)，CPU 要 90 秒

#### onnx 佔用 GPU 暫時難以解決

專案中所使用的 OCR 模型是用 .onnx 檔，而無論模型多大， `ONNXRuntime` 為了效能，會在啟動前先劃分一個很大的空間給模型。</br>
可以透過以下 code 關閉此機制，但可能造成效能下降，甚至卡住不動。

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.enable_mem_pattern = False
so.enable_mem_reuse = False
session = ort.InferenceSession('your_model.onnx', providers=['CUDAExecutionProvider'], sess_options=so)
```

- Note: PyTorch 是 lazy allocation、用多少分多少、動態增減，ONNXRuntime 傾向分一大塊做池，方便快取與併發。
- Reference: [[Performance] Find out why the GPU memory allocated with CUDAExecutionProvider is much larger than the ONNX size #14526](https://github.com/microsoft/onnxruntime/issues/14526)

#### RapidOCR 無法手動限制 GPU 用量

當前版本是使用 rapidocr_onnxruntime，此模型有2個缺點

1. 沒有開放上一節中方法的接口，你要做只能修改 RapidOCR 的 source code
2. rapidocr_onnxruntime 對 GPU 的支援更差([Link](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr/install/#_2))，建議改用 rapidocr_paddle 。很不幸，docling 是用 rapidocr_onnxruntime。

```text
请使用Python3.6及以上版本。
rapidocr_onnxruntime系列库目前仅在CPU上支持较好，GPU上推理很慢，这一点可参考link。因此不建议用onnxruntime-gpu版推理。
GPU端推理推荐用rapidocr_paddle
```

Note: Docling 呼叫 RapidOCR 的程式碼在 `docling.models.rapid_ocr_model` Line 45，你可以通過修改 Line 54 `use_cuda = False` 來強迫 RapidOCR 用 CPU

## 路線圖（Roadmap）

- 索引/載入器 CLI 化與批次化
- 回答品質評估（faithfulness/answer relevancy）流程
- 多來源（非 GoodNotes）的載入器與格式化
- 改善 OCR/DET/REC 的 GPU 記憶體佔用策略 -->
