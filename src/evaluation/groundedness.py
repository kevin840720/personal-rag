"""
@File    :  groundedness.py
@Time    :  2025/09/22 00:00:00
@Author  :  Kevin Wang
@Desc    :  以 LLM 評估回答是否扎根於檢索內容。

Metrics:
以 LLM 評估回答品質，有兩項指標：
    - ResponseGroundedness：檢驗回答(response)是否引用了文本內容(retrieved_contexts)。評分只有 0.0/0.5/1.0 三檔，只需要 response、retrieved_contexts
    - Faithfulness：將回答(response)逐句檢驗是否能在文本(retrieved_contexts)中找到依據，最終計算「有依據」句子的比例作為分數。需要 user_input、response、retrieved_contexts。

備註：
    - ResponseGroundedness 與 Faithfulness 都是用於檢驗幻覺，但 ResponseGroundedness 看中整體的語意表達，Faithfulness 則是逐句檢視
"""

from typing import List
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.metrics import (ResponseGroundedness,
                           Faithfulness,
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

class GroundednessEval:
    def __init__(self,
                 model:str="gpt-4o-mini",
                 api_key:str=os.getenv("OPEN_AI_API"),
                 max_workers:int=4,
                 ):
        """初始化檢驗 LLM 回覆與 context 的評估器。

        Args:
            model: 用於評估的 `ChatOpenAI` 模型名稱。
            api_key: OpenAI API 金鑰，預設讀取 `OPEN_AI_API`。

        備註:
            - 此評估器包含兩個指標：`ResponseGroundedness`、`Faithfulness`。
            - 以 LLM 判斷回答是否被檢索內容支撐，溫度固定為 0.0 以提高一致性。
        """
        self._llm = ChatOpenAI(model=model,
                               api_key=api_key,
                               temperature=0.0,
                               )
        self._evaluator_llm = LangchainLLMWrapper(self._llm)
        self._metrics = [ResponseGroundedness(llm=self._evaluator_llm),
                         Faithfulness(llm=self._evaluator_llm),
                         ]
        self._max_workers = max_workers

    def _check_dataset(self, dataset:EvaluationDataset) -> None:
        """檢查資料集欄位是否完整。

        Args:
            dataset: Ragas 的 `EvaluationDataset`，每筆需包含必要欄位。

        Raises:
            EvaluationDatasetError: 當資料集為空、載入失敗，或樣本缺少 `user_input`、`response`、`retrieved_contexts`。
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
            if not sample.user_input:
                raise EvaluationDatasetError(f"row {idx} missing 'user_input'")
            if not (isinstance(sample.response, str) and sample.response.strip()):
                raise EvaluationDatasetError(f"row {idx} missing 'response'")
            rc = sample.retrieved_contexts
            if not isinstance(rc, list) or len(rc) == 0:
                raise EvaluationDatasetError(f"row {idx} 'retrieved_contexts' must be non-empty list")

    def evaluate(self,
                 dataset:EvaluationDataset,
                 ) -> EvaluationResult:
        """執行 groundedness 與 faithfulness 評估。

        Args:
            dataset: 經由 `EvaluationDataset` 封裝的資料集。

        Raises:
            EvaluationDatasetError: 若資料集欄位檢查未通過。

        Returns:
            EvaluationResult: Ragas 的評估結果物件，可透過 `.to_pandas()` 取得 DataFrame。
        """
        self._check_dataset(dataset)
        results = evaluate(dataset=dataset,
                           metrics=self._metrics,
                           raise_exceptions=True,
                           run_config=RunConfig(max_workers=self._max_workers),
                           )
        return results

class GroundednessEvalWithoutReference(GroundednessEval):
    pass