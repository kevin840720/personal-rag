# personal-rag

面向「私人筆記（目前只處理日文筆記）」的檢索增強生成（RAG）系統。

本專案包含：

- MCP 工具伺服器：提供檢索工具（向量 + 關鍵字 BM25）給外部 Agent 使用
- 以 OpenAI 相容介面對接 OpenWebUI
- GoodNotes PDF/圖片 OCR 與筆記抽取管線：將手寫筆記與掃描教材轉為可檢索文本

整體串接：（OpenAPI 工具）↔ OpenWebUI ↔ MCP Server

## 功能特色

- 向量檢索 + 關鍵字檢索：
  - 向量：PostgreSQL + pgvector（`text-embedding-3-small`，附 Redis 嵌入快取）
  - 關鍵字：Elasticsearch BM25
- MCP 工具：`search_japanese_note` 同時支援語意（query）與關鍵字（keywords）檢索，並合併去重
- GoodNotes 文件處理：黑底手寫與白底掃描皆支援，含自動前處理、偵測、辨識、分群與主幹文字萃取
- OpenWebUI 整合：
  - 以 OpenAPI「工具伺服器」引入 MCP 工具
- 評估支持：提供 RAGAS 檢索評估腳本，便於觀察檢索純度與覆蓋率

## 啟動 MCP Server 並接到 Open WebUI

### 需求

- 已安裝 `pipenv` 中套件
- 已安裝 `mcpo`：`pipenv install mcpo`
- Open WebUI（具「Add Tool Server / OpenAPI」功能）
- OpenAI API Key (用於設定 Open WebUI)

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

## 環境與服務

- 主要環境變數（可置於 `.env` 或 shell）：
  - OpenAI：`OPEN_AI_API`
  - Redis：`MY_REDIS_HOST`、`MY_REDIS_PORT`、`MY_REDIS_PASSWORD`
  - PostgreSQL/pgvector：`MY_POSTGRE_HOST`、`MY_POSTGRE_PORT`、`MY_POSTGRE_DB_NAME`、`MY_POSTGRE_USERNAME`、`MY_POSTGRE_PASSWORD`
  - Elasticsearch：`MY_ELASTIC_HOST`、`MY_ELASTIC_PORT`、`MY_ELASTIC_USERNAME`、`MY_ELASTIC_PASSWORD`
  - 其他：`MCP_SERVICE_URL`、`VOLUME_PATH`

- 以 Docker 啟動基礎服務：

```bash
docker compose up -d
```

請先設定 `.env` 中的對應變數（特別是 `VOLUME_PATH` 與各服務連線資訊）。

## 文件載入與 GoodNotes 支援

- 針對 GoodNotes 的 PDF/圖片，提供完整 OCR 與主幹文字萃取管線：
  - 黑底手寫與白底掃描皆支援，多種前處理（反白、去色、加強灰階、顏色過濾等）
  - 偵測（DET）→ 切塊 → 辨識（REC）→ 鄰近群組合併 → 主幹文字輸出
  - 取得每頁 metadata（頁碼、書籤 outlines 等）
- 快速上手與範例程式請見：`src/ingestion/file_loaders/goodnotes/readme.md`

## Project Organization

```text
personal-rag/
├── Pipfile                           # pipenv 依賴設定
├── Pipfile.lock                      # 依賴版本鎖定
├── README.md                         # 專案介紹與啟動方式
├── __version__.py                    # 專案版本資訊
├── conftest.py                       # pytest 共用設定
├── docker-compose.yaml               # Docker 服務編排設定
├── docker-start-up.sh                # 啟動 Docker 並初始化資料夾的腳本
├── models/                           # 模型相關資源
│   └── ocr/
│       └── PPv5-download-guide.md    # OCR 模型下載說明
├── notebooks/                        # Jupyter 筆記本範例
│   ├── example.ipynb                 # 基本示範
│   ├── file_loader.ipynb             # 檔案載入流程示例
│   └── goodnotes.ipynb               # Goodnotes 筆記處理範例
└── src/                              # 主要程式碼
    ├── base.py                       # 預留基底模組
    ├── cache/                        # 快取模組
    │   ├── base.py                   # 快取處理器介面
    │   ├── errors.py                 # 快取相關錯誤定義
    │   └── redis.py                  # Redis 快取實作
    ├── chunking/                     # 分塊處理
    │   ├── base.py                   # 分塊處理器介面
    │   └── docling.py                # Docling 分塊與合併邏輯
    ├── embedding/                    # 向量嵌入
    │   ├── base.py                   # Embedding 模型介面
    │   └── openai_embed.py           # OpenAI Embedding 實作含快取
    ├── infra/                        # 儲存與基礎設施
    │   ├── base.py                   # 預留基底模組
    │   └── stores/                   # 索引儲存實作
    │       ├── base.py               # 儲存介面與 SearchHit 定義
    │       ├── elasticsearch.py      # Elasticsearch BM25/向量索引
    │       ├── errors.py             # 儲存層錯誤類型
    │       └── pgvector.py           # PostgreSQL pgvector 儲存實作
    ├── ingestion/                    # 文件載入器
    │   ├── base.py                   # Loader 抽象類與結果模型
    │   ├── file_loaders/             # 各式檔案載入器
    │   │   ├── goodnotes.py          # Goodnotes 輸出的 PDF/圖片載入
    │   │   └── file_loaders/goodnotes/  # GoodNotes OCR 管線與工具
    │   │       ├── loader.py         # 端到端載入器（可插拔 DET/REC）
    │   │       ├── pipeline.py       # 背景判定、前處理、偵測/辨識、分群
    │   │       ├── ops.py            # 影像處理原子操作
    │   │       └── readme.md         # GoodNotes 流程與使用說明
    │   │   ├── image.py              # 影像載入與 OCR
    │   │   ├── markdown.py           # Markdown 載入
    │   │   ├── office_docx.py        # DOCX 載入
    │   │   ├── office_excel.py       # Excel 載入
    │   │   └── pdf.py                # PDF 載入與 OCR/表格處理
    │   └── utils.py                  # Docling 序列化與表格工具
    ├── mcp_server/                   # MCP 相關服務
    │   ├── jp_learning_rag.py        # 日文學習筆記 RAG MCP server
    │   └── manager.py                # MCP 工具註冊與權限管理
    └── objects.py                    # Document/Chunk 等核心資料型別
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
