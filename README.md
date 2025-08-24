# personal-rag

用於檢索私人日文筆記的 RAG 系統。  
透過 MCP <-> Open WebUI <-> OpenAI API 串接運行。
本專案只負責 MCP 的部分。

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
    ├── app_mcp.py                    # FastAPI MCP 工具服務
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
    │   │   ├── image.py              # 影像載入與 OCR
    │   │   ├── markdown.py           # Markdown 載入
    │   │   ├── office_docx.py        # DOCX 載入
    │   │   ├── office_excel.py       # Excel 載入（讀值模式）
    │   │   └── pdf.py                # PDF 載入與 OCR/表格處理
    │   └── utils.py                  # Docling 序列化與表格工具
    ├── mcp_server/                   # MCP 相關服務
    │   ├── jp_learning_rag.py        # 日文學習筆記 RAG MCP server
    │   └── manager.py                # MCP 工具註冊與權限管理
    ├── objects.py                    # Document/Chunk 等核心資料型別
    └── react.py                      # LangChain ReACT Agent
```
