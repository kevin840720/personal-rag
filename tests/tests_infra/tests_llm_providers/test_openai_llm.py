# -*- encoding: utf-8 -*-
"""
@File    :  test_openai_llm.py
@Time    :  2025/08/23 10:43:16
@Author  :  Kevin Wang
@Desc    :  None
"""

# test_openai_llm_provider_real.py
import os
import pytest
import asyncio

from infra.llm_providers.base import ChatMessage, LLMConfig
from infra.llm_providers.openai_llm import OpenAILLMProvider
from infra.llm_providers.errors import RateLimitError, AuthenticationError, LLMProviderError

from conftest import SKIP_TESTS_USE_EXTERNAL_API_TESTS

@pytest.fixture
def dummy_config():
    return LLMConfig(model_name="gpt-4.1-nano",  # 最便宜的版本，GPT-5 是 reasoning model ，別被騙了
                     temperature=0.1,
                     max_tokens=16,
                     service_tier="standard"
                     )

@pytest.fixture
def dummy_messages():
    return [ChatMessage(role="user", content="你是誰？")]

@pytest.fixture
def provider(dummy_config):
    api_key = os.environ.get("OPEN_AI_API")
    assert api_key is not None, "請設定 OPEN_AI_API 環境變數"
    return OpenAILLMProvider(config=dummy_config,
                             api_key=api_key,
                             base_url=None,  # 官方就用預設
                             )

@pytest.mark.skipif(SKIP_TESTS_USE_EXTERNAL_API_TESTS, reason="Skipping OpenAI API tests")
class TestOpenAILLMProvider:
    @pytest.mark.asyncio
    async def test_chat_completion(self, provider, dummy_messages):
        reply = await provider.chat_completion(dummy_messages)
        assert isinstance(reply, type(dummy_messages[0]))
        assert reply.role == "assistant"
        assert reply.content and isinstance(reply.content, str)

    @pytest.mark.asyncio
    async def test_chat_completion_stream(self, provider, dummy_messages):
        out = []
        async for chunk in provider.chat_completion_stream(dummy_messages):
            assert isinstance(chunk, str)
            out.append(chunk)
        assert "".join(out)
        assert len(out) >= 2  # 確認 stream 功能有在運作

    @pytest.mark.asyncio
    async def test_health_check_real(self, provider):
        result = await provider.health_check()
        assert result is True

