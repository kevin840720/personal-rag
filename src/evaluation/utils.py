
from typing import List, Dict, Any

from ragas import EvaluationDataset, SingleTurnSample
import json


class EvalTools:
    @classmethod
    def _load_jsonl(cls, path:str) -> List[Dict[str,Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
        
    
    @classmethod
    def get_eval_dataset(cls, path:str)->EvaluationDataset:
        rows = cls._load_jsonl(path)
        samples:List[SingleTurnSample] = []
        for row in rows:
            samples.append(
                SingleTurnSample(
                    user_input=row.get("user_input"),                  # 使用者提問 (Query, Q)
                    retrieved_contexts=row.get("retrieved_contexts"),  # 檢索器取回的內容 (Context, C)
                    response=row.get("response"),                      # LLM 輸出的最終回答 (Answer, A)
                    reference=row.get("reference"),                    # 標準答案 (Gold Answer，用於 A vs Gold)
                    reference_contexts=row.get("reference_contexts"),  # 標註的正確文件 (Gold Context，用於檢索評估)
                )
            )
        return EvaluationDataset(samples=samples)

    @classmethod
    def get_eval_dataset_from_rows(cls, rows:List[Dict[str,Any]])->EvaluationDataset:
        samples:List[SingleTurnSample] = []
        for row in rows:
            samples.append(
                SingleTurnSample(
                    user_input=row["user_input"],
                    retrieved_contexts=row["retrieved_contexts"],
                    response=row.get("response"),
                    reference=row.get("reference"),
                    reference_contexts=row.get("reference_contexts"),
                )
            )
        return EvaluationDataset(samples=samples)
