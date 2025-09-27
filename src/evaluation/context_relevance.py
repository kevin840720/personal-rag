"""
@File    :  context_relevance.py
@Time    :  2025/09/10 19:27:08
@Author  :  Kevin Wang
@Desc    :  評估檢索階段品質。

Metrics:
    - ContextRelevance：檢驗回傳的文本內容(retrieved_contexts)是否貼合提問(query)。評分只有 0.0/0.5/1.0 三檔，只需要 user_input、retrieved_contexts。
    - LLMContextPrecisionWithReference：對每個 retrieved_context，用 LLM 判斷該 retrieved_contexts 是否支撐 reference，最後計算平均。需要 user_input、retrieved_contexts、reference。
    - LLMContextPrecisionWithoutReference（未使用）：用 response 取代 LLMContextPrecisionWithReference 中的 reference。需要 user_input、retrieved_contexts、response。（我覺得沒啥用）
    - LLMContextRecall：將 reference 拆成數個 statements，再用 LLM 判斷哪些敘述能被 retrieved_contexts 支撐。需要 user_input、retrieved_contexts、reference。
"""

from typing import List
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.metrics import (ContextRelevance,
                           LLMContextPrecisionWithReference,
                           LLMContextRecall,
                           )
from ragas import (EvaluationDataset,
                   evaluate,
                   )
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import (EvaluationResult,
                                  SingleTurnSample,
                                  )

from evaluation.errors import EvaluationDatasetError
from logging_.logger import get_logger


logger = get_logger(__name__)

load_dotenv()

class ContextRelevanceEval:
    def __init__(self,
                 model:str="gpt-4o-mini",
                 api_key:str=os.getenv("OPEN_AI_API"),
                 max_workers:int=4,
                 ):
        """初始化具 reference 的檢索評估器。

        Args:
            model: 用於評估的 `ChatOpenAI` 模型名稱。
            api_key: OpenAI API 金鑰，預設讀取 `OPEN_AI_API`。

        備註:
            - 此評估器包含三個指標：`LLMContextPrecisionWithReference`、`LLMContextRecall`、`ContextRelevance`。
            - 為了結果穩定性，溫度固定為 0.0。
        """
        self._llm = ChatOpenAI(model=model,
                               api_key=api_key,
                               temperature=0.0,
                               )
        self._evaluator_llm = LangchainLLMWrapper(self._llm)
        self._metrics = [LLMContextPrecisionWithReference(llm=self._evaluator_llm),  # precision@k: 找回來的內容有多少是真正有用的？
                         LLMContextRecall(llm=self._evaluator_llm),                  # recall@k:    應該找回來的東西裡，有多少真的被找回來了？
                         ContextRelevance(llm=self._evaluator_llm),                  # 檢索結果到底有沒有真的回應使用者的問題？
                         ]
        self._max_workers = max_workers

    def _check_dataset(self, dataset:EvaluationDataset) -> None:
        """檢查資料集欄位是否完整（具 reference 版本）。

        此方法會逐筆檢查並在第一筆錯誤即拋出例外，不返回布林值。

        Args:
            dataset: Ragas 的 `EvaluationDataset`，每筆需包含最小欄位。

        Raises:
            EvaluationDatasetError: 當資料集載入失敗、為空、或任一筆樣本缺以下欄位時：
                - `user_input` 缺失或為空。
                - `retrieved_contexts` 非非空的 List。
                - `reference` 缺失或為空字串。
        """
        try:
            samples:List[SingleTurnSample] = dataset.samples
        except Exception:
            logger.warning("Dataset load fail")
            raise EvaluationDatasetError("dataset load failed")
        if not samples:
            logger.warning("Empty Dataset")
            raise EvaluationDatasetError("empty dataset")
        for idx, sample in enumerate(samples):
            # 檢查資料集是否完整，應包含
            #   - user_input：使用者輸入
            #   - retrieved_contexts：透過 RAG 獲得的文件
            #   - reference(*註)：標準答案，應包含回答使用者輸入所需要的所有資訊
            if not sample.user_input:
                raise EvaluationDatasetError(f"row {idx} missing 'user_input'")
            rc = sample.retrieved_contexts
            if not isinstance(rc, list) or len(rc) == 0:
                raise EvaluationDatasetError(f"row {idx} 'retrieved_contexts' must be non-empty list")
            # 註：這裡的 reference 並非 precision/recall 理論上必須的元素，而是一種實務上的簡化與折衷：
            # 透過人為撰寫答案 (reference)，再由 LLM 判斷 context 是否支撐答案，
            # 以避免需要對所有 retrieved_contexts (Chunk) 逐一人工標註「是否相關」。
            if not (isinstance(sample.reference, str) and sample.reference.strip()):
                raise EvaluationDatasetError(f"row {idx} missing 'reference'")

    def evaluate(self, dataset:EvaluationDataset) -> EvaluationResult:
        """執行三項檢索評估（具 reference 版本）。

        Args:
            dataset: 經由 `EvaluationDataset` 封裝的資料集。

        Raises:
            EvaluationDatasetError: 若資料集欄位檢查未通過。

        回傳:
            None。函式會直接 `print(results.to_pandas())` 以便觀察結果。
        """
        self._check_dataset(dataset)
        results = evaluate(dataset=dataset,
                           metrics=self._metrics,
                           raise_exceptions=True,
                           run_config=RunConfig(max_workers=self._max_workers),
                           )
        return results


class ContextRelevanceEvalWithoutReference:
    """適用於線上 RAG 流程的評估器，無需事先準備標準答案（reference/reference_contexts）。"""
    def __init__(self,
                 model:str="gpt-4o-mini",
                 api_key:str=os.getenv("OPEN_AI_API"),
                 max_workers:int=4,
                 ):
        """初始化無 reference 的檢索評估器（適合線上 RAG 串接）。

        Args:
            model: 用於評估的 `ChatOpenAI` 模型名稱。
            api_key: OpenAI API 金鑰，預設讀取 `OPEN_AI_API`。

        備註:
            - 此評估器包含一個指標：`ContextRelevance`。
        """
        self._llm = ChatOpenAI(model=model,
                               api_key=api_key,
                               temperature=0.0,
                               )
        self._evaluator_llm = LangchainLLMWrapper(self._llm)
        self._metrics = [ContextRelevance(llm=self._evaluator_llm),                  # 檢索結果到底有沒有真的回應使用者的問題？
                         ]
        self._max_workers = max_workers

    def _check_dataset(self, dataset:EvaluationDataset) -> None:
        """檢查資料集欄位是否完整（無 reference 版本）。

        此方法會逐筆檢查並在第一筆錯誤即拋出例外，不返回布林值。

        Args:
            dataset: Ragas 的 `EvaluationDataset`，每筆需包含最小欄位。

        Raises:
            EvaluationDatasetError: 當資料集載入失敗、為空、或任一筆樣本缺以下欄位時：
                - `user_input` 缺失或為空。
                - `retrieved_contexts` 非空的 List。
        """
        try:
            samples:List[SingleTurnSample] = dataset.samples
        except Exception:
            logger.warning("Dataset load fail")
            raise EvaluationDatasetError("dataset load failed")
        if not samples:
            logger.warning("Empty Dataset")
            raise EvaluationDatasetError("empty dataset")
        for idx, sample in enumerate(samples):
            # 檢查資料集是否完整，應包含
            #   - user_input：使用者輸入
            #   - retrieved_contexts：透過 RAG 獲得的文件
            if not sample.user_input:
                raise EvaluationDatasetError(f"row {idx} missing 'user_input'")
            rc = sample.retrieved_contexts
            if not isinstance(rc, list) or len(rc) == 0:
                raise EvaluationDatasetError(f"row {idx} 'retrieved_contexts' must be non-empty list")

    def evaluate(self, dataset:EvaluationDataset) -> EvaluationResult:
        """執行兩項檢索評估（無 reference 版本）。

        Args:
            dataset: 經由 `EvaluationDataset` 封裝的資料集。

        Raises:
            EvaluationDatasetError: 若資料集欄位檢查未通過。

        回傳:
            None。函式會直接 `print(results.to_pandas())` 以便觀察結果。
        """
        self._check_dataset(dataset)
        results = evaluate(dataset=dataset,
                           metrics=self._metrics,
                           raise_exceptions=True,
                           run_config=RunConfig(max_workers=self._max_workers),
                           )
        return results
