
from typing import List, Dict, Any
import json
import logging


from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from tenacity import (Retrying,
                      WrappedFn,
                      after_log,
                      retry_if_exception_type,
                      stop_after_attempt,
                      wait_random_exponential,
                      )
from tenacity.after import after_nothing


class RagasPatch:
    @classmethod
    def add_retry(cls,
                  fn:WrappedFn,
                  run_config:RunConfig,
                  ) -> WrappedFn:
        """
        Adds retry functionality to a given function using the provided RunConfig.

        This function wraps the input function with retry logic using the tenacity library.
        It configures the retry behavior based on the settings in the RunConfig.

        Notes
        -----
        - If log_tenacity is enabled in the RunConfig, it sets up logging for retry attempts.
        - The retry logic uses exponential backoff with random jitter for wait times.
        - The number of retry attempts and exception types to retry on are configured
        based on the RunConfig.
        """
        # configure tenacity's after section wtih logger
        if run_config.log_tenacity is not None:
            logger = logging.getLogger(f"ragas.retry.{fn.__name__}")
            tenacity_logger = after_log(logger, logging.DEBUG)
        else:
            tenacity_logger = after_nothing

        r = Retrying(
            wait=wait_random_exponential(multiplier=2, max=run_config.max_wait),
            stop=stop_after_attempt(run_config.max_retries),
            retry=retry_if_exception_type(run_config.exception_types),
            reraise=True,
            after=tenacity_logger,
        )
        return r.wraps(fn)

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
