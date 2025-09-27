"""
@File    :  answer_relevance.py
@Time    :  2025/09/22 00:00:00
@Author  :  Kevin Wang
@Desc    :  


Metrics:
以 LLM 評估回答品質，有四項指標：
    - AnswerAccuracy：檢查「回答跟參考答案是不是同一件事」，LLM 雙向互評判斷。評分只有 0/0.5/1 三檔，需要 user_input、response、reference。
    - AnswerCorrectness：把參考答案與系統回答拆成句子，先判斷每句是否被支撐（F1，預設權重 0.75），再加上一個語意相似度分數（預設 0.25）。比 AnswerAccuracy 更複雜的判斷方式，需要 user_input、response、reference。
    - AnswerRelevancy：用 LLM 從回答生成多個問題，再計算這些問題與原始問題的 embedding 餘弦相似度並取平均。但若回答(response)帶有閃避或不承諾語氣，直接以零分計算。只需要 user_input、response
    - AnswerSimilarity：純語意相似度比較，衡量回答與參考答案的向量距離。只需要 response、reference。

AnswerCorrectness 指標公式：

    AnswerCorrectness = w1 * Fβ(factuality) + w2 * S(similarity)

    其中：
    - factuality:
        1. 將 response 與 reference 各自拆解成陳述句集合 A = {a1, a2, ...}, G = {g1, g2, ...}
        2. 用 LLM (CorrectnessClassifier) 判斷 ai 中每句是否為 gj 的支持句
        3. 統計 TP / FP / FN
            - TP = | { ai ∈ A | ∃ gj ∈ G, ai 被判斷為 gj 的支持句 } |
            - FP = | { ai ∈ A | ∀ gj ∈ G, ai 未被判斷為支持句 } |
            - FN = | { gj ∈ G | ∀ ai ∈ A, gj 未被任何回答句支持 } |
        4. 套入 Fβ 公式:
            - precision = TP/(TP + FP)
            -   recall  = TP/(TP + FN)
            - Fβ = (1 + β²) * Precision * Recall / (β² * Precision + Recall)
                (Harmonic mean of Precision and Recall with weight ratio β²:1)
    - S = 語義相似度 ∈ [0,1]，由 embeddings 計算
    - 預設 w1=0.75, w2=0.25, β=1

備註：
    - (v0.3.4) AnswerRelevancy 有潛在的坑：
        - RAGAS 有兩個 Embeddings 基類：BaseRagasEmbedding、BaseRagasEmbeddings，兩者 interface 不完全相同
        - AnswerRelevancy 的運算是基於 BaseRagasEmbeddings 設計的（需要 embed_query 方法），但大多數 ragas.embeddings 的方法是繼承自 BaseRagasEmbedding
        - 官方維護者直接把對應 issue(#529) 關掉並標記「not planned」，未來將以 BaseRagasEmbedding 為主，並移除 BaseRagasEmbeddings
"""

from typing import List
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from ragas.metrics import (AnswerAccuracy,
                           AnswerCorrectness,
                           AnswerRelevancy,
                           AnswerSimilarity,
                           )
from ragas import (EvaluationDataset,
                   evaluate,
                   )
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import (EvaluationResult,
                                  SingleTurnSample,
                                  )
from ragas.embeddings import OpenAIEmbeddings

from evaluation.errors import EvaluationDatasetError
from logging_.logger import get_logger


logger = get_logger(__name__)

load_dotenv()


class AnswerRelevanceEval:
    def __init__(self,
                 model:str="gpt-4o-mini",
                 api_key:str=os.getenv("OPEN_AI_API"),
                 embedding_model:str="text-embedding-3-small",
                 max_workers:int=4,
                 ):
        """初始化回答品質評估器。

        Args:
            model: 用於評估的 `ChatOpenAI` 模型名稱。
            api_key: OpenAI API 金鑰，預設讀取 `OPEN_AI_API`。
            embedding_model: OpenAI embedding 模型名稱，供 AnswerCorrectness / AnswerRelevancy / AnswerSimilarity 使用。
            max_workers: Ragas 併發執行緒數。

        備註:
            - 指標組合：`AnswerAccuracy`（LLM 對答案斷詞比對）、`AnswerCorrectness`（0.75*F1 + 0.25*Semantic Similarity）、
              `AnswerRelevancy`（回答是否回應原問題）、`AnswerSimilarity`（純語意相似度）。
            - 所有指標均需 `response` 與 `reference`，部分額外依賴 embeddings。
        """
        self._llm = ChatOpenAI(model=model,
                               api_key=api_key,
                               temperature=0.0,
                               )
        self._evaluator_llm = LangchainLLMWrapper(self._llm)
        client = AsyncOpenAI(api_key=api_key)
        self._embeddings = OpenAIEmbeddings(client=client,
                                            model=embedding_model,
                                            )
        self._metrics = [AnswerAccuracy(llm=self._evaluator_llm),
                         AnswerCorrectness(llm=self._evaluator_llm,
                                           embeddings=self._embeddings,
                                           ),
                        # NOTE: 社群預計用 BaseRagasEmbedding 取代 BaseRagasEmbeddings，但目前 AnswerRelevancy 是仍使用 BaseRagasEmbeddings，會導致報錯。
                        #  AnswerRelevancy(llm=self._evaluator_llm,
                        #                  embeddings=self._embeddings,
                        #                  ),
                         AnswerSimilarity(embeddings=self._embeddings),
                         ]
        self._max_workers = max_workers

    def _check_dataset(self, dataset:EvaluationDataset) -> None:
        """檢查資料集是否具備回答評估所需欄位。"""
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
            if not (isinstance(sample.reference, str) and sample.reference.strip()):
                raise EvaluationDatasetError(f"row {idx} missing 'reference'")

    def evaluate(self,
                 dataset:EvaluationDataset,
                 ) -> EvaluationResult:
        """執行回答品質評估並回傳 Ragas `EvaluationResult`。"""
        self._check_dataset(dataset)
        results = evaluate(dataset=dataset,
                           metrics=self._metrics,
                           raise_exceptions=True,
                           run_config=RunConfig(max_workers=self._max_workers),
                           )
        return results

# 唯一不要求 reference 的指標 AnswerRelevancy 會報錯，先停用
# class AnswerRelevanceEvalWithoutReference:
#     def __init__(self,
#                  model:str="gpt-4o-mini",
#                  api_key:str=os.getenv("OPEN_AI_API"),
#                  embedding_model:str="text-embedding-3-small",
#                  max_workers:int=4,
#                  ):
#         """僅使用 AnswerRelevancy 評估回答（無 reference）。

#         Args:
#             model: 用於評估的 `ChatOpenAI` 模型名稱。
#             api_key: OpenAI API 金鑰，預設讀取 `OPEN_AI_API`。
#             embedding_model: OpenAI embedding 模型名稱，供 AnswerRelevancy 使用。
#             max_workers: Ragas 併發執行緒數。
#         """
#         self._llm = ChatOpenAI(model=model,
#                                api_key=api_key,
#                                temperature=0.0,
#                                )
#         self._evaluator_llm = LangchainLLMWrapper(self._llm)
#         client = AsyncOpenAI(api_key=api_key)
#         self._embeddings = OpenAIEmbeddings(client=client,
#                                             model=embedding_model,
#                                             )
#         self._metrics = [# NOTE: 社群預計用 BaseRagasEmbedding 取代 BaseRagasEmbeddings，但目前 AnswerRelevancy 是仍使用 BaseRagasEmbeddings，會導致報錯。
#                          #  AnswerRelevancy(llm=self._evaluator_llm,
#                          #                  embeddings=self._embeddings,
#                          #                  )
#                          ]
#         self._max_workers = max_workers

#     def _check_dataset(self, dataset:EvaluationDataset) -> None:
#         """檢查資料集欄位是否完整（無 reference 版本）。"""
#         try:
#             samples:List[SingleTurnSample] = dataset.samples
#         except Exception:
#             logger.warning("Dataset load fail")
#             raise EvaluationDatasetError("dataset load failed")
#         if not samples:
#             logger.warning("Empty Dataset")
#             raise EvaluationDatasetError("empty dataset")
#         for idx, sample in enumerate(samples):
#             if not sample.user_input:
#                 raise EvaluationDatasetError(f"row {idx} missing 'user_input'")
#             if not (isinstance(sample.response, str) and sample.response.strip()):
#                 raise EvaluationDatasetError(f"row {idx} missing 'response'")

#     def evaluate(self,
#                  dataset:EvaluationDataset,
#                  ) -> EvaluationResult:
#         """執行 AnswerRelevancy 評估（無 reference 版本）。"""
#         self._check_dataset(dataset)
#         results = evaluate(dataset=dataset,
#                            metrics=self._metrics,
#                            raise_exceptions=True,
#                            run_config=RunConfig(max_workers=self._max_workers),
#                            )
#         return results
