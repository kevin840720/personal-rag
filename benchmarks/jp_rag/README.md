# JP RAG Benchmarks (Reference-based)

本目錄提供 JP RAG 的線上基準評估（reference-based, online 檢索）。不與 `tests/` 混在一起，直接以腳本執行，沿用 `src/evaluation` 工具。

- 資料集：`datasets/jp_reference.jsonl`（JSONL，每行一筆樣本）
- 執行腳本：`reference_eval/run_reference_eval.py`

執行前準備：
- 於專案根目錄建立/設定 `.env`，並確保 OpenAI API 與向量/全文檢索所需連線設定存在（參考 `src/mcp_server/jp_learning_rag.py`）。
- 安裝相依套件（已在 Pipfile 中）。

執行方式（擇一）：
- 建議：設定 `PYTHONPATH=src` 後執行：
  - `PYTHONPATH=src python benchmarks/jp_rag/reference_eval/run_reference_eval.py`

你需要填寫/調整的地方：
- `datasets/jp_reference.jsonl`：每行一筆 JSON，欄位如下：
  - `query`: 使用者問題（str，必填）
  - `keywords`: 關鍵字（List[str]，可為空；建議 2–6 個中/日文對應詞）
  - `reference`: 參考答案（str，必填）
- `reference_eval/run_reference_eval.py`：如需改模型，調整 `MODEL_NAME`；如需更換資料集，調整 `DATASET_PATH`。此腳本改用 `JapaneseLearningRAGServer.retrieve_chunks` 取得 `List[Chunk]`，並以 `ch.content` 形成 `retrieved_contexts`；不解析工具的文字輸出。
