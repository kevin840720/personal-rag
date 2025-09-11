"""
@File    :  run_reference_eval.py
@Time    :  2025/09/10 00:00:00
@Author  :  Kevin Wang
@Desc    :  JP RAG 線上（Online）基準評估。透過 JapaneseLearningRAGServer 的 `retrieve_chunks` 檢索，
           用 `ContextRelevanceEval` 執行評測，並輸出逐列結果 CSV 與整體/分組平均的 JSON meta。

設計準則（必讀）：
    - 檢索入口唯一：僅呼叫 `retrieve_chunks`（不啟動 MCP），同程序 direct-call 減少變因；本腳本自行從 Chunk 取 `content`。
    - 評測入口唯一：僅呼叫 `ContextRelevanceEval.evaluate(dataset)`，不得使用任何私有屬性或子指標。
    - 場景固定三類：embedding（query 有、keywords 空）、keywords（query 空、keywords 有）、mix（兩者皆有）。
    - Context 來源契約：本腳本使用 `retrieve_chunks` 回傳之 `List[Chunk]`，並以 `ch.content` 構成 `retrieved_contexts`；不解析文字輸出。
    - I/O 收斂：輸出一個 CSV（逐列結果）與一個 JSON（含 overall 與 by-type averages）。

資料集格式（每行一筆 JSON）：
    - query:     str，使用者問題
    - keywords:  List[str]，可為空，固定中/日對應詞（如「奧運」「オリンピック」）
    - reference: str，標準答案

執行說明：
    - 需在 .env 設定 OPEN_AI_API（沿用現有程式命名）。
    - 執行：PYTHONPATH=src pipenv run python benchmarks/jp_rag/reference_eval/run_reference_eval.py

輸出說明：
    - CSV：每筆原始資料會展開為 3 列（embedding/keywords/mix）。
    - JSON：
        - rows_original：原始資料筆數（未展開）。
        - rows_expanded：展開後的列數（= rows_original * 3）。
        - averages / averages_by_type：僅對數值指標欄位取平均（排除 line_no 等非指標欄）。

介面相依：
    - `ContextRelevanceEval.evaluate` 必須回傳帶有 `.to_pandas()` 的結果物件（EvaluationResult）。若非此行為，腳本會直接失敗。
"""

from pathlib import Path
import asyncio

import json
from datetime import datetime
from typing import Generator,Any,List,Tuple,TypedDict,Literal

from dotenv import load_dotenv


from evaluation.utils import EvalTools
from evaluation.context_relevance import ContextRelevanceEval
from mcp_server.jp_learning_rag import JapaneseLearningRAGServer
import pandas as pd
load_dotenv()



class MetaRow(TypedDict):
    """Meta 列描述。

    Attributes:
        line_no: 原始資料集的行號（1-based）。
        eval_type: 評測情境類型，embedding/keywords/mix。
    """
    line_no:int
    eval_type:Literal['embedding','keywords','mix']


class EvalRow(TypedDict):
    """評測輸入資料列。

    Attributes:
        user_input: 使用者問題（Query）。
        retrieved_contexts: 檢索得到的 contexts 內容清單。
        reference: 人工撰寫的參考答案（Gold Answer）。
    """
    user_input:str
    retrieved_contexts:List[str]
    reference:str

class ReferenceEvalRunner:
    """JP RAG 參考基準（reference-based）評測執行器。

    將資料建構、評測、聚合與輸出分步執行，避免巨型 `main`。

    Args:
        model_name: 用於 Ragas 評測的 OpenAI 模型名稱。
        dataset_path: JSONL 資料集路徑。
        top_k: 檢索的 embedding 與 keyword 上限（同值使用）。
        report_dir: 報表輸出目錄。
    """
    def __init__(self,
                 model_name:str="gpt-4o-mini",
                 dataset_path:Path=Path(__file__).resolve().parents[1] / "datasets" / "jp_reference.jsonl",
                 top_k:int=3,
                 report_dir:Path=Path(__file__).resolve().parents[1] / "report",
                 eval_max_workers:int=4,
                 ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.report_dir = report_dir
        self.eval_max_workers = eval_max_workers

    def _load_jsonl(self, path:str) -> Generator[dict, Any, Any]:
        """讀取並驗證 JSONL 資料集。

        Args:
            path: JSONL 檔案路徑字串。

        Yields:
            dict: 每行的 JSON 資料，並附加 `line_no` 欄位。

        Raises:
            ValueError: 若缺少必需欄位 `query`/`reference`，或 `keywords` 類型不正確。
        """
        with open(path, "r", encoding="utf-8") as file:
            idx = 0
            for line in file:
                idx += 1
                line = line.strip()
                if not line:
                    continue

                data:dict = json.loads(line)
                if not (isinstance(data.get("query"), str) and data.get("query").strip()):
                    raise ValueError(f"row {idx} missing 'query'")
                kws = data.get("keywords") or []
                if not isinstance(kws, list) or any(not isinstance(kw, str) for kw in kws):
                    raise ValueError(f"row {idx} 'keywords' must be List[str] or omitted")
                if any(not kw.strip() for kw in kws):
                    raise ValueError(f"row {idx} 'keywords' must not contain empty strings")
                ref = data.get("reference")
                if not (isinstance(ref, str) and ref.strip()):
                    raise ValueError(f"row {idx} missing 'reference'")
                
                data['line_no'] = idx
                yield data

    async def _build_rows(self) -> Tuple[List[MetaRow],List[EvalRow]]:
        """建立評測所需的資料列（Meta 與 Eval）。

        針對資料集中每筆樣本，併發進行三種檢索情境並抽取 `content` 作為 contexts。

        Returns:
            Tuple[List[MetaRow], List[EvalRow]]: 評測的 meta 列與資料列。
        """
        server = JapaneseLearningRAGServer()
        built_metas:List[MetaRow] = []
        built_dataset:List[EvalRow] = []
        for data in self._load_jsonl(str(self.dataset_path)):
            emb_chunks, kw_chunks, mix_chunks = await asyncio.gather(
                server.retrieve_chunks(query=data['query'],
                                       keywords=[],
                                       top_embedding_k=self.top_k,
                                       top_keyword_k=0,
                                       ),
                server.retrieve_chunks(query="",
                                       keywords=data['keywords'],
                                       top_embedding_k=0,
                                       top_keyword_k=self.top_k,
                                       ),
                server.retrieve_chunks(query=data['query'],
                                       keywords=data['keywords'],
                                       top_embedding_k=self.top_k,
                                       top_keyword_k=self.top_k,
                                       ),
            )

            built_metas.append({"line_no": data['line_no'], "eval_type":  "embedding"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": [ch.content for ch in emb_chunks],
                                  "reference": data['reference'],
                                  })

            built_metas.append({"line_no": data['line_no'], "eval_type":  "keywords"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": [ch.content for ch in kw_chunks],
                                  "reference": data['reference'],
                                  })

            built_metas.append({"line_no": data['line_no'], "eval_type":  "mix"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": [ch.content for ch in mix_chunks],
                                  "reference": data['reference'],
                                  })
        return built_metas, built_dataset

    def _evaluate(self, rows:List[EvalRow]) -> pd.DataFrame:
        """以 Ragas 指標執行評測並回傳 DataFrame 結果。

        Args:
            rows: 已展開且含 contexts 的評測資料列。

        Returns:
            pd.DataFrame: Ragas `EvaluationResult.to_pandas()` 的表格結果。
        """
        dataset = EvalTools.get_eval_dataset_from_rows(rows)
        evaluator = ContextRelevanceEval(model=self.model_name,
                                         max_workers=self.eval_max_workers,
                                         )
        results = evaluator.evaluate(dataset=dataset)
        return results.to_pandas()

    def _aggregate(self,
                   metas:List[MetaRow],
                   df_data:pd.DataFrame,
                   ) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,List[str]]:
        """合併結果並計算整體與分組平均。

        Args:
            metas: 評測 meta 列（含 `line_no` 與 `eval_type`）。
            df_data: 指標結果的 DataFrame 資料。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]: 依序為合併表、整體平均、分組平均、數值欄位名單。
        """
        df_meta = pd.DataFrame(metas)
        df = pd.concat([df_meta, df_data], axis=1)
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in ("line_no",)]
        avg_df:pd.DataFrame = pd.DataFrame([df[numeric_cols].mean(numeric_only=True)])
        group_avg_df:pd.DataFrame = df.groupby("eval_type")[numeric_cols].mean().reset_index()
        return df, avg_df, group_avg_df, numeric_cols

    def _save_reports(self,
                      df:pd.DataFrame,
                      avg_df:pd.DataFrame,
                      group_avg_df:pd.DataFrame,
                      numeric_cols:List[str],
                      metas:List[MetaRow],
                      ) -> None:
        """輸出 CSV 與 JSON 報表。

        Args:
            df: 合併後的逐列結果。
            avg_df: 整體平均的單列 DataFrame。
            group_avg_df: 依 `eval_type` 分組的平均。
            numeric_cols: 參與平均計算的數值欄位。
            metas: Meta 列資料（用於 rows_original 計算）。
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        ts:str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_csv = self.report_dir / f"eval_results_{ts}.csv"
        meta_json = self.report_dir / f"eval_meta_{ts}.json"
        df.to_csv(result_csv, index=False)

        averages = {k: float(v) for k, v in avg_df.iloc[0].to_dict().items()}
        averages_by_type = {}
        for _, row in group_avg_df.iterrows():
            averages_by_type[row["eval_type"]] = {k: float(row[k]) for k in numeric_cols}
        rows_expanded = len(df)
        rows_original = len({m["line_no"] for m in metas})
        with open(meta_json, "w", encoding="utf-8") as file:
            json.dump({"model": self.model_name,
                       "dataset": str(self.dataset_path),
                       "top_k": self.top_k,
                       "rows_original": rows_original,
                       "rows_expanded": rows_expanded,
                       "result_csv": str(result_csv),
                       "averages": averages,
                       "averages_by_type": averages_by_type,
                       }, file, ensure_ascii=False, indent=4)
        print(f"[INFO] reports saved: {result_csv} and {meta_json}")

    def run(self) -> None:
        """執行完整評測流程並輸出結果。"""
        metas, rows = asyncio.run(self._build_rows())
        df_data = self._evaluate(rows)
        df, avg_df, group_avg_df, numeric_cols = self._aggregate(metas, df_data)
        print("[INFO] metric averages:")
        print(avg_df)
        print("[INFO] metric averages by eval_type:")
        print(group_avg_df)
        self._save_reports(df, avg_df, group_avg_df, numeric_cols, metas)


if __name__ == "__main__":
    runner = ReferenceEvalRunner()
    runner.run()
