# -*- encoding: utf-8 -*-
"""
@File    :  test_groundedness.py
@Time    :  2025/09/23 00:00:00
@Author  :  Kevin Wang
@Desc    :  Integration tests for Groundedness evaluators (real API, no mocks)
"""

from typing import List

import pytest
from dotenv import load_dotenv

from conftest import SKIP_TESTS_USE_EXTERNAL_API_TESTS
from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset

from evaluation.groundedness import GroundednessEval


load_dotenv()


@pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping external API tests")
def test_groundedness_with_reference_runs():
    """最小可運行：具 reference 的 groundedness 評估可成功執行。"""
    samples:List[SingleTurnSample] = [
        SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            response="The Eiffel Tower is located in Paris.",
            retrieved_contexts=["The Eiffel Tower is located in Paris."],
            reference="The Eiffel Tower is located in Paris.",
        )
    ]
    dataset = EvaluationDataset(samples=samples)

    evaluator = GroundednessEval()
    evaluator.evaluate(dataset)

