# -*- encoding: utf-8 -*-
"""
@File    :  test_openai_embed.py
@Time    :  2025/08/21 07:09:40
@Author  :  Kevin Wang
@Desc    :  None
"""


from typing import List
import os

import pytest

from cache.redis import RedisCacheHandler
from conftest import (SKIP_REDIS_TESTS,
                      SKIP_TESTS_USE_EXTERNAL_API_TESTS,
                      )
from embedding.openai_embed import (EmbeddingError,
                                    OpenAIEmbeddingModel,
                                    )

@pytest.fixture
def model() -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(api_key=os.getenv("OPEN_AI_API"),
                                model_name="text-embedding-3-small",
                                max_retries=3,
                                memory_cache=None,
                                )

@pytest.fixture
def model_w_cache() -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(api_key=os.getenv("OPEN_AI_API"),
                                model_name="text-embedding-3-small",
                                max_retries=3,
                                memory_cache=RedisCacheHandler(host=os.getenv("MY_REDIS_HOST"),
                                                               port=os.getenv("MY_REDIS_PORT"),
                                                               password=os.getenv("MY_REDIS_PASSWORD"),
                                                               ),
                                )

@pytest.fixture
def sample_text() -> str:
    return "人工智慧是一門跨學科的科學，涉及數學、統計學等多個領域。"

@pytest.fixture
def sample_texts() -> List[str]:
    return ["是以古之作者，寄身於翰墨，見意於篇籍，不假良史之辭，不託飛馳之勢，而聲名自傳於後。",
            "To be, or not to be, that is the question.",
            "なんで春日影やったの！？",
            "這句話是工人智慧寫的。",
            "這句話不是人工智慧寫的。",
            ]

class TestOpenAIEmbeddingModel:
    @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API test")
    def test_encode(self,
                    model:OpenAIEmbeddingModel,
                    sample_text:str,
                    ):
        """Test if the model successfully encodes text."""
        embedding = model.encode(sample_text)
        assert isinstance(embedding, list), "Embedding should be a list."
        assert all(isinstance(value, float) for value in embedding), "All values should be floats."
        assert len(embedding) == 1536, "text-embedding-3-small is of length 1536"

    def test_encode_empty_text_error(self,
                                     model:OpenAIEmbeddingModel,
                                     ):
        with pytest.raises(EmbeddingError, match="Input text cannot be empty."):
            model.encode("")

    def test_encode_illegal_model(self,
                                  model:OpenAIEmbeddingModel,
                                  ):
        model.model_name = "invalid-model-name"
        with pytest.raises(EmbeddingError):
            model.encode("This should fail.")

    @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API test")
    def test_encode_batch(self,
                          model:OpenAIEmbeddingModel,
                          sample_texts:List[str],
                          ):
        """Test batch encoding for multiple texts."""
        embeddings = model.encode_batch(sample_texts)
        assert len(embeddings) == len(sample_texts), "Number of embeddings should match input texts."
        assert all(isinstance(embedding, list) for embedding in embeddings), "Each embedding should be a list."
        assert all(len(embedding) == 1536 for embedding in embeddings), "text-embedding-3-small is of length 1536"

    @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API test")
    def test_similarity(self,
                        model:OpenAIEmbeddingModel,
                        sample_text:str,
                        sample_texts:List[str],
                        ):
        """Test similarity computation between texts."""
        scores = [model.similarity(sample_text, text) for text in sample_texts]
        assert len(scores) == len(sample_texts), "Number of scores should match number of candidates."
        assert all(-1 <= score <= 1 for score in scores), "Similarity scores should be between -1 and 1."

    @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API test")
    def test_most_similar(self,
                          model:OpenAIEmbeddingModel,
                          sample_text:str,
                          sample_texts:List[str],
                          ):
        """Test retrieving the most similar text."""
        results = model.most_similar(sample_text, sample_texts, top_k=2)
        assert len(results) == 2, "Should return top 2 most similar texts."
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results), \
            "Each result should be a tuple (text, similarity_score)."
        assert all(-1 <= result[1] <= 1 for result in results), "Similarity scores should be between -1 and 1."
        assert results[0][0] == '這句話是工人智慧寫的。'  # Embedding 說這句話比較相近，cosine similarity 約是 0.44
        assert results[1][0] == '這句話不是人工智慧寫的。'  # cosine similarity 約是 0.42

    @pytest.mark.skipif(SKIP_REDIS_TESTS, reason="Skipping OpenAI API test")
    @pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API test")
    def test_cache_usage(self,
                         model_w_cache:OpenAIEmbeddingModel,
                         ):
        """Test if the caching mechanism works correctly."""
        import uuid
        text = str(uuid.uuid4()) 
        try:
            embedding_1 = model_w_cache.encode(text)
            cached_embedding = model_w_cache._load_cache(text)
            assert cached_embedding is not None, "Embedding should be cached after first call."
            assert cached_embedding == embedding_1, "Cached embedding should match the first generated embedding."
        finally:
            model_w_cache.memory_cache.delete(text)
