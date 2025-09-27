"""
@File    :  run_reference_eval.py
@Time    :  2025/09/10 00:00:00
@Author  :  Kevin Wang
@Desc    :  
"""

from datetime import datetime
from pathlib import Path
from typing import (Any,
                    Dict,
                    Generator,
                    Iterable,
                    List,
                    Literal,
                    Optional,
                    Tuple,
                    TypedDict,
                    )
import asyncio
import json
import os
import warnings

from dotenv import load_dotenv
from langchain_core.messages import (HumanMessage,
                                     SystemMessage,
                                     )
from langchain_openai import ChatOpenAI
import numpy as np
import pandas as pd

from evaluation.answer_relevance import AnswerRelevanceEval
from evaluation.context_relevance import ContextRelevanceEval
from evaluation.groundedness import GroundednessEval
from evaluation.utils import EvalTools
from monitoring.langfuse_client import get_langfuse_client, observe_if_enabled
from mcp_server.jp_learning_rag import JapaneseLearningRAGServer


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
    response:str


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
                 dataset_path:Path=Path(__file__).resolve().parents[0] / "datasets" / "jp_reference.jsonl",
                 top_k:int=3,
                 report_dir:Path=Path(__file__).resolve().parents[0] / "report",
                 eval_max_workers:int=4,
                 answer_model:str="gpt-4o-mini",
                 answer_temperature:float=0.0,
                 ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.report_dir = report_dir
        self.eval_max_workers = eval_max_workers
        self._langfuse = get_langfuse_client()
        self._langfuse_tags:List[str] = ["jp_rag", "evaluation"]
        self._answer_llm = ChatOpenAI(model=answer_model,
                                      api_key=os.getenv("OPEN_AI_API"),
                                      temperature=answer_temperature,
                                      )
        self._context_eval = ContextRelevanceEval(model=self.model_name,
                                                  max_workers=self.eval_max_workers,
                                                  )
        self._answer_eval = AnswerRelevanceEval(model=self.model_name,
                                                max_workers=self.eval_max_workers,
                                                )
        self._ground_eval = GroundednessEval(model=self.model_name,
                                             max_workers=self.eval_max_workers,
                                             )

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

            # 確保資料庫不為空，且 Retriever 能正常運作
            if len(emb_chunks) == 0:
                raise RuntimeError(f"Vector database retrieved no chunk with query: {data['query']}")
            if len(kw_chunks) == 0:
                raise RuntimeError(f"Lexical database retrieved no chunk with keywords: {data['keywords']}")
            if len(mix_chunks) == 0:
                raise RuntimeError("Both vector and lexical database retrieved no chunk."
                                   f" query: {data['query']} & keywords: {data['keywords']}"
                                   )
            
            # 若檢索到空內容的 Chunk，則只需要 warning，流程繼續
            for ch in emb_chunks + kw_chunks + mix_chunks:
                if not ch.content:
                    warnings.warn(f"Chunk {ch.id} is with empty content")

            emb_contexts = [ch.content for ch in emb_chunks]
            kw_contexts = [ch.content for ch in kw_chunks]
            mix_contexts = [ch.content for ch in mix_chunks]

            emb_response = await self._generate_response(query=data['query'],
                                                         contexts=emb_contexts,
                                                         )
            kw_response = await self._generate_response(query=data['query'],
                                                        contexts=kw_contexts,
                                                        )
            mix_response = await self._generate_response(query=data['query'],
                                                         contexts=mix_contexts,
                                                         )

            built_metas.append({"line_no": data['line_no'], "eval_type":  "embedding"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": emb_contexts,
                                  "reference": data['reference'],
                                  "response": emb_response,
                                  })

            built_metas.append({"line_no": data['line_no'], "eval_type":  "keywords"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": kw_contexts,
                                  "reference": data['reference'],
                                  "response": kw_response,
                                  })

            built_metas.append({"line_no": data['line_no'], "eval_type":  "mix"})
            built_dataset.append({"user_input": data['query'],
                                  "retrieved_contexts": mix_contexts,
                                  "reference": data['reference'],
                                  "response": mix_response,
                                  })
        return built_metas, built_dataset

    async def _generate_response(self,
                                 *,
                                 query:str,
                                 contexts:List[str],
                                 ) -> str:
        """使用檢索 contexts 生成回覆。"""
        joined_context = "\n\n".join(f"[Context {idx}]\n{ctx}" for idx, ctx in enumerate(contexts, start=1))
        messages = [SystemMessage(content="你是日文學習助教，需根據提供的內容回答問題，不能編造。"),
                    HumanMessage(content=(f"問題：{query}\n\n"
                                          f"檢索內容：\n{joined_context}\n\n"
                                          "請用繁體中文回答，若資訊不足請坦承不知道。"))]
        reply = await self._answer_llm.ainvoke(messages)
        content = reply.content
        if isinstance(content, list):
            text = "".join(part for part in content if isinstance(part, str))
        else:
            text = str(content)
        return text.strip()

    def _evaluate_context(self, rows:List[EvalRow]) -> pd.DataFrame:
        """以檢索品質指標執行評測。（query ↔ retrieved context）"""
        dataset = EvalTools.get_eval_dataset_from_rows(rows)
        results = self._context_eval.evaluate(dataset=dataset)
        return results.to_pandas()

    def _evaluate_answers(self, rows:List[EvalRow]) -> pd.DataFrame:
        """以回答品質指標執行評測。（query ↔ answer）"""
        dataset = EvalTools.get_eval_dataset_from_rows(rows)
        results = self._answer_eval.evaluate(dataset=dataset)
        return results.to_pandas()

    def _evaluate_groundedness(self, rows:List[EvalRow]) -> pd.DataFrame:
        """以 groundedness 指標執行評測。（retrieved context ↔ answer）"""
        dataset = EvalTools.get_eval_dataset_from_rows(rows)
        results = self._ground_eval.evaluate(dataset=dataset)
        return results.to_pandas()

    def _merge_evaluations(self,
                           context_df:pd.DataFrame,
                           answer_df:pd.DataFrame,
                           grounded_df:pd.DataFrame,
                           ) -> pd.DataFrame:
        """整併不同評測結果。"""
        answer_metrics = answer_df.drop(columns=[c for c in ("user_input",
                                                             "retrieved_contexts",
                                                             "reference",
                                                             ) if c in answer_df.columns],
                                         errors="ignore",
                                         )
        grounded_metrics = grounded_df.drop(columns=[c for c in ("user_input",
                                                                 "retrieved_contexts",
                                                                 "reference",
                                                                 "response",
                                                                 ) if c in grounded_df.columns],
                                             errors="ignore",
                                             )
        combined = pd.concat([context_df.reset_index(drop=True),
                              answer_metrics.reset_index(drop=True),
                              grounded_metrics.reset_index(drop=True),
                              ], axis=1)
        return combined

    def _aggregate(self,
                   metas:List[MetaRow],
                   df_data:pd.DataFrame,
                   ) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,List[str],dict[str,int]]:
        """合併結果並計算整體與分組平均。

        Args:
            metas: 評測 meta 列（含 `line_no` 與 `eval_type`）。
            df_data: 指標結果的 DataFrame 資料。

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], dict[str,int]]: 依序為合併表、整體平均、分組平均、數值欄位名單、各數值欄位參與樣本數。
        """
        df_meta = pd.DataFrame(metas)
        df = pd.concat([df_meta, df_data], axis=1)
        numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in ("line_no",)]
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
        valid_counts_series = numeric_df.notna().sum()
        avg_df:pd.DataFrame = pd.DataFrame([numeric_df.mean(numeric_only=True)])
        group_numeric = numeric_df.copy()
        group_numeric["eval_type"] = df["eval_type"]
        group_avg_df:pd.DataFrame = group_numeric.groupby("eval_type", dropna=False).mean(numeric_only=True).reset_index()
        return df, avg_df, group_avg_df, numeric_cols, {k:int(v) for k, v in valid_counts_series.to_dict().items()}

    def _save_reports(self,
                      df:pd.DataFrame,
                      averages:Dict[str,Optional[float]],
                      averages_by_type:Dict[str,Dict[str,Optional[float]]],
                      valid_counts:dict[str,int],
                      timestamp:str,
                      rows_original:int,
                      rows_expanded:int,
                      run_id:str,
                      ) -> Tuple[Path,Path]:
        """輸出 CSV 與 JSON 報表，並回傳路徑。

        Args:
            df: 合併後的逐列結果。
            averages: 指標整體平均的字典表示。
            averages_by_type: 依 `eval_type` 分組的平均數據。
            valid_counts: 每個指標的有效樣本數。
            timestamp: 報表檔案名稱所用的時間戳字串。
            rows_original: 原始樣本數。
            rows_expanded: 展開後的樣本數。
            run_id: 評估批次的唯一識別字串。

        Returns:
            Tuple[Path, Path]: 依序為結果 CSV 及 Meta JSON 的路徑。
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        result_csv = self.report_dir / f"eval_results_{timestamp}.csv"
        meta_json = self.report_dir / f"eval_meta_{timestamp}.json"
        df.to_csv(result_csv, index=False)

        with open(meta_json, "w", encoding="utf-8") as file:
            json.dump({"model": self.model_name,
                       "dataset": str(self.dataset_path),
                       "top_k": self.top_k,
                       "rows_original": rows_original,
                       "rows_expanded": rows_expanded,
                       "result_csv": str(result_csv),
                       "averages": averages,
                       "averages_by_type": averages_by_type,
                       "valid_counts": {k: int(v) for k, v in valid_counts.items()},
                       "run_id": run_id,
                       "report_timestamp": timestamp,
                       }, file, ensure_ascii=False, indent=4)
        print(f"[INFO] reports saved: {result_csv} and {meta_json}")
        return result_csv, meta_json

    def _publish_langfuse(self,
                          *,
                          run_id:str,
                          timestamp:str,
                          averages:Dict[str,Optional[float]],
                          averages_by_type:Dict[str,Dict[str,Optional[float]]],
                          valid_counts:dict[str,int],
                          rows_original:int,
                          rows_expanded:int,
                          result_csv:Path,
                          meta_json:Path,
                          ) -> None:
        """將評估摘要寫入 Langfuse。"""
        client = self._langfuse
        if client is None:
            return
        try:
            with client.start_as_current_span(name="jp_reference_eval",
                                              input={"model": self.model_name,
                                                     "dataset": str(self.dataset_path),
                                                     "top_k": self.top_k,
                                                     "rows_original": rows_original,
                                                     "rows_expanded": rows_expanded,
                                                     "report_csv": str(result_csv),
                                                     "report_meta": str(meta_json),
                                                     },
                                              metadata={"report_timestamp": timestamp},
                                              ) as span:
                span.update_trace(name=run_id,
                                  tags=self._langfuse_tags,
                                  metadata={"report_timestamp": timestamp,
                                            "run_id": run_id,
                                            },
                                  )
                for metric, value in averages.items():
                    if value is None:
                        continue
                    span.score(name=f"avg.{metric}", value=float(value))
                for eval_type, metrics in averages_by_type.items():
                    for metric, value in metrics.items():
                        if value is None:
                            continue
                        span.score(name=f"{eval_type}.{metric}", value=float(value))
                span.update(output={"averages": averages,
                                    "averages_by_type": averages_by_type,
                                    "valid_counts": valid_counts,
                                    "rows_original": rows_original,
                                    "rows_expanded": rows_expanded,
                                    })
            client.flush()
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to publish Langfuse trace: {exc}")

    @observe_if_enabled(name="jp_reference_eval.run",
                        capture_input=False,
                        capture_output=False,
                        )
    def run(self) -> None:
        """執行完整評測流程並輸出結果。"""
        metas, rows = asyncio.run(self._build_rows())
        context_df = self._evaluate_context(rows)
        answer_df = self._evaluate_answers(rows)
        grounded_df = self._evaluate_groundedness(rows)
        df_data = self._merge_evaluations(context_df,
                                          answer_df,
                                          grounded_df,
                                          )
        df, avg_df, group_avg_df, numeric_cols, valid_counts = self._aggregate(metas, df_data)
        averages:Dict[str,Optional[float]] = {}
        if not avg_df.empty:
            first_row = avg_df.iloc[0]
            for col in numeric_cols:
                value = first_row.get(col)
                averages[col] = float(value) if pd.notna(value) else None
        averages_by_type:Dict[str,Dict[str,Optional[float]]] = {}
        for _, row in group_avg_df.iterrows():
            eval_type = row["eval_type"]
            averages_by_type[eval_type] = {}
            for col in numeric_cols:
                value = row.get(col)
                averages_by_type[eval_type][col] = float(value) if pd.notna(value) else None
        rows_expanded = len(df)
        rows_original = len({m["line_no"] for m in metas})
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        run_id = f"jp_reference_eval_{timestamp}"
        print("[INFO] metric averages:")
        print(avg_df)
        print("[INFO] metric averages by eval_type:")
        print(group_avg_df)
        result_csv, meta_json = self._save_reports(df,
                                                   averages,
                                                   averages_by_type,
                                                   valid_counts,
                                                   timestamp,
                                                   rows_original,
                                                   rows_expanded,
                                                   run_id,
                                                   )
        self._publish_langfuse(run_id=run_id,
                               timestamp=timestamp,
                               averages=averages,
                               averages_by_type=averages_by_type,
                               valid_counts=valid_counts,
                               rows_original=rows_original,
                               rows_expanded=rows_expanded,
                               result_csv=result_csv,
                               meta_json=meta_json,
                               )


if __name__ == "__main__":
    runner = ReferenceEvalRunner()
    runner.run()
