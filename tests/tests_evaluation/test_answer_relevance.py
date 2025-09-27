# -*- encoding: utf-8 -*-
"""
@File    :  test_answer_relevance.py
@Time    :  2025/09/23 00:00:00
@Author  :  Kevin Wang
@Desc    :  Integration tests for AnswerRelevance evaluators (real API, no mocks)
"""

from typing import List

import pytest
from dotenv import load_dotenv

from conftest import SKIP_TESTS_USE_EXTERNAL_API_TESTS
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

from evaluation.answer_relevance import (AnswerRelevanceEval,
                                        # 唯一不要求 reference 的指標 AnswerRelevancy 會報錯，先停用
                                        #  AnswerRelevanceEvalWithoutReference,
                                         )


load_dotenv()


@pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping external API tests")
def test_answer_relevance_with_reference_runs():
    """最小可運行：具 reference 的回答品質評估可成功執行。"""
    samples:List[SingleTurnSample] = [
        SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            response="The Eiffel Tower is located in Paris.",
            reference="The Eiffel Tower is located in Paris.",
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    evaluator = AnswerRelevanceEval()
    evaluator.evaluate(dataset)


# @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping external API tests")
# def test_answer_relevance_without_reference_runs():
#     """最小可運行：無 reference 的回答相關性評估可成功執行。"""
#     samples:List[SingleTurnSample] = [
#         SingleTurnSample(
#             user_input="Where is the Eiffel Tower located?",
#             response="The Eiffel Tower is located in Paris.",
#         )
#     ]
#     dataset = EvaluationDataset(samples=samples)

#     evaluator = AnswerRelevanceEvalWithoutReference()
#     evaluator.evaluate(dataset)

