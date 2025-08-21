# -*- encoding: utf-8 -*-
"""
@File    :  test_openai_embed.py
@Time    :  2025/08/21 07:09:40
@Author  :  Kevin Wang
@Desc    :  None
"""

from typing import List
from unittest.mock import Mock
import math

import pytest

from embedding.openai_embed import (EmbeddingError,
                                    OpenAIEmbeddingModel,
                                    )

@pytest.fixture
def model() -> OpenAIEmbeddingModel:
    # 不依賴真金鑰；大多數測試會 monkeypatch client 或 encode
    return OpenAIEmbeddingModel(api_key="ThisAPIKeyIsFake",
                                model_name="text-embedding-3-small",
                                max_retries=3,
                                memory_cache=None,
                                )

@pytest.fixture
def sample_text() -> str:
    return "人工智慧是一門跨學科的科學，涉及數學、統計學等多個領域。"

def _make_client_mock(mocker, vectors:List[List[str]]) -> Mock:
    """
    回傳一個 mock 物件，模擬 openai 官方 client(openai.OpenAI) 的 embeddings.create(CreateEmbeddingResponse) 方法行為，  
    主要用於測試 encode(text:str) -> List[float] 時對 openai embeddings API 的呼叫。

    """
    embeddings = mocker.Mock()
    it = iter(vectors)
    # 模擬 openai 回傳結構: response.data[0].embedding
    def _create(input, model):
        data0 = mocker.Mock()
        data0.embedding = next(it)
        response = mocker.Mock()
        response.data = [data0]
        return response
    embeddings.create.side_effect = _create
    client = mocker.Mock()
    client.embeddings = embeddings
    return client

class TestOpenAIEmbeddingModel:
    def test_encode(self,
                    model:OpenAIEmbeddingModel,
                    sample_text,
                    mocker,
                    ):
        dim = 1536  # Same as text-embedding-3-small
        client = _make_client_mock(mocker, [[0.1]*dim])
        # 替換實例屬性
        mocker.patch.object(model, "client", client)
        vec = model.encode(sample_text)
        assert isinstance(vec, list)
        assert len(vec) == dim
        assert all(isinstance(x, float) for x in vec)
        client.embeddings.create.assert_called_once()

    def test_encode_empty_text_error(self,
                                     model:OpenAIEmbeddingModel,
                                     ):
        with pytest.raises(EmbeddingError, match="Input text cannot be empty."):
            model.encode("")

    def test_encode_retries_error(self,
                                  model:OpenAIEmbeddingModel,
                                  sample_text:str,
                                  mocker,  # Built-in pytest fixture
                                  ):
        # 每次呼叫都丟例外 → 走完整 retry 並轉為 EmbeddingError
        broken_embeddings = mocker.Mock()
        broken_embeddings.create.side_effect = RuntimeError("boom")
        broken_client = mocker.Mock(embeddings=broken_embeddings)
        mocker.patch.object(model, "client", broken_client)

        with pytest.raises(EmbeddingError, match="failed after"):
            model.encode(sample_text)
        assert broken_embeddings.create.call_count == model.max_retries

    def test_encode_batch(self,
                          model:OpenAIEmbeddingModel,
                          mocker,  # Built-in pytest fixture
                          ):
        client = _make_client_mock(mocker,
                                   [[1.0 if i == k else 0.0 for i in range(1536)] for k in range(10)]
                                   )
        mocker.patch.object(model, "client", client)

        texts = [f"encoded string {_}" for _ in range(10)]
        vectors = model.encode_batch(texts)

        # 檢驗長度
        assert len(vectors) == len(texts)
        assert all([len(_) == 1536 for _ in vectors])
        assert client.embeddings.create.call_count == len(texts)

        # 檢驗我 Mock 沒寫錯，每個 vector 都不一樣，且只有一個分量是1
        assert len({tuple(_) for _ in vectors}) == len(texts)
        for _ in vectors:
            assert sum(_) == 1

    def test_similarity(self,
                        model:OpenAIEmbeddingModel,
                        mocker,  # Built-in pytest fixture
                        ):
        def dummy_encode(text: str):
            mapping = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
            return mapping[text]

        mocker.patch.object(model, "encode", side_effect=dummy_encode)
        ab = model.similarity("a", "b")
        ac = model.similarity("a", "c")
        cc = model.similarity("c", "c")

        assert math.isclose(ab, 0.0, abs_tol=1e-8)
        assert math.isclose(ac, 1 / math.sqrt(2), rel_tol=1e-8)
        assert math.isclose(cc, 1.0, rel_tol=1e-8)


    def test_most_similar(self,
                          model:OpenAIEmbeddingModel,
                          mocker,  # Built-in pytest fixture
                          ):
        def fake_encode(text: str):
            mapping = {"q": [1.0, 0.0],
                       "v1": [0.9, 0.1],
                       "v2": [0.0, 1.0],
                       "v3": [0.7, 0.3],
                       }
            return mapping[text]

        mocker.patch.object(model, "encode", side_effect=fake_encode)
        candidates = ["v1", "v2", "v3"]
        top2 = model.most_similar("q", candidates, top_k=2)
        assert [t for t, _ in top2] == ["v1", "v3"]
        assert all(0.0 <= s <= 1.0 for _, s in top2)

    def test_dim(self,
                 model:OpenAIEmbeddingModel,
                 mocker,  # Built-in pytest fixture
                 ):
        mocker.patch.object(model, "encode", return_value=[0.0] * 1536)
        assert model.dim == 1536


#     def test_cache_usage_prevents_api_call(mocker):
#         # 簡單快取
#         class DummyCache:
#             def __init__(self):
#                 self.store = {}
#             def get(self, k):
#                 return self.store.get(k)
#             def set(self, k, v):
#                 self.store[k] = v

#         cache = DummyCache()
#         text = "cache me"
#         key = hashlib.md5(text.encode()).hexdigest()
#         cached_vec = [0.42] * 1536
#         cache.set(key, cached_vec)

#         m = OpenAIEmbeddingModel(api_key="DUMMY", memory_cache=cache)

#         # 若 encode 命中快取，就不應呼叫 API
#         embeddings = mocker.Mock()
#         embeddings.create.side_effect = AssertionError("Should not call API when cached")
#         client = mocker.Mock(embeddings=embeddings)
#         mocker.patch.object(m, "client", client)

#         got = m.encode(text)
#         assert got == cached_vec
#         embeddings.create.assert_not_called()


# # ===== Optional: real API tests (skipped by default) =====

# SKIP_REAL = os.getenv("SKIP_TESTS_USE_EXTERNAL_API_TESTS", "1") != "0"

# @pytest.mark.skipif(SKIP_REAL, reason="Skipping OpenAI real API call")
# def test_real_encode_dimension(model, sample_text):
#     vec = model.encode(sample_text)
#     assert isinstance(vec, list)
#     assert all(isinstance(x, float) for x in vec)
#     assert len(vec) == 1536