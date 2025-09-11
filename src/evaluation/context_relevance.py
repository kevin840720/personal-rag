"""
@File    :  context_relevance.py
@Time    :  2025/09/10 19:27:08
@Author  :  Kevin Wang
@Desc    :  https://docs.ragas.io/en/v0.3.2/howtos/integrations/griptape/?h=contextrelevance#evaluating-retrieval
"""

from typing import List
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.metrics import (ContextRelevance,
                           LLMContextPrecisionWithReference,
                           LLMContextPrecisionWithoutReference,
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
from evaluation.utils import RagasPatch
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
            - 此評估器包含兩個指標：`ContextRelevance`、`LLMContextPrecisionWithoutReference`。
            - 以 LLM 判斷 context 是否支撐系統當前 `response`，不依賴人工黃金標註。
        """
        self._llm = ChatOpenAI(model=model,
                               api_key=api_key,
                               temperature=0.0,
                               )
        self._evaluator_llm = LangchainLLMWrapper(self._llm)
        self._metrics = [ContextRelevance(llm=self._evaluator_llm),                     # 檢索結果到底有沒有真的回應使用者的問題？
                         LLMContextPrecisionWithoutReference(llm=self._evaluator_llm),  # 使用用系統 response 當 proxy，LLM 判斷 context 是否支撐 response。
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
                - `retrieved_contexts` 非非空的 List。
                - `response` 缺失或為空字串。
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
            #   - response：LLM 回應
            if not sample.user_input:
                raise EvaluationDatasetError(f"row {idx} missing 'user_input'")
            rc = sample.retrieved_contexts
            if not isinstance(rc, list) or len(rc) == 0:
                raise EvaluationDatasetError(f"row {idx} 'retrieved_contexts' must be non-empty list")
            if not (isinstance(sample.response, str) and sample.response.strip()):
                raise EvaluationDatasetError(f"row {idx} missing 'response'")

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
