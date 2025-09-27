# -*- encoding: utf-8 -*-
"""
@File    :  test_context_relevance.py
@Time    :  2025/09/10 00:00:00
@Author  :  Kevin Wang
@Desc    :  Integration tests for ContextRelevance evaluators (real API, no mocks)
"""

from typing import List
import os

import pytest
from dotenv import load_dotenv

from conftest import SKIP_TESTS_USE_EXTERNAL_API_TESTS
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

from evaluation.context_relevance import (ContextRelevanceEval,
                                          ContextRelevanceEvalWithoutReference, 
                                          )


load_dotenv()


@pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping external API tests")
def test_eval_with_reference_runs():
    """最小可運行：具 reference 的檢索評估流程能成功執行。"""
    samples:List[SingleTurnSample] = [
        SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            retrieved_contexts=["The Eiffel Tower is located in Paris."],
            reference="The Eiffel Tower is located in Paris.",
            # response="The Eiffel Tower is located in Paris.",
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    evaluator = ContextRelevanceEval()
    evaluator.evaluate(dataset)


@pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping external API tests")
def test_eval_without_reference_runs():
    """最小可運行：無 reference（線上 RAG 串接）評估流程能成功執行。"""
    samples:List[SingleTurnSample] = [
        SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            retrieved_contexts=["The Eiffel Tower is located in Paris."],
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    evaluator = ContextRelevanceEvalWithoutReference()
    evaluator.evaluate(dataset)

