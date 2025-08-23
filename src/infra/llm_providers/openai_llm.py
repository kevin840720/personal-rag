# -*- encoding: utf-8 -*-
"""
@File    :  openai_llm.py
@Time    :  2025/08/23 10:21:22
@Author  :  Kevin Wang
@Desc    :  None
"""

import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Literal
from openai import AsyncOpenAI
import json

from infra.llm_providers.base import AbstractLLMProvider, ChatMessage, LLMConfig
from infra.llm_providers.errors import LLMProviderError, RateLimitError, AuthenticationError

class OpenAIReasoningConfig(LLMConfig):
    service_tier:Literal["standard", "flex", "priority"]="standard"

class OpenAILLMProvider(AbstractLLMProvider):
    def __init__(self,
                 config:LLMConfig,
                 api_key:str,
                 base_url:Optional[str]=None,
                 ):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key,
                                  base_url=base_url,
                                  )

    def _convert_messages(self,
                          messages:List[ChatMessage],
                          ) -> List[Dict[str,str]]:
        """轉換內部訊息格式為 OpenAI 格式"""
        return [{"role": msg.role, "content": msg.content}
                for msg in messages
                ]
    
    async def chat_completion(self,
                              messages:List[ChatMessage],
                              **kwargs,
                              ) -> ChatMessage:
        """非串流聊天完成"""
        try:
            openai_messages = self._convert_messages(messages)
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
                **kwargs
            )
            
            content = response.choices[0].message.content
            return ChatMessage(role="assistant",
                               content=content,
                               metadata={
                                   "model": self.config.model_name,
                                   "usage": response.usage.model_dump() if response.usage else None,
                                   "finish_reason": response.choices[0].finish_reason
                               })
            
        except Exception as err:
            await self._handle_error(err)
    
    async def chat_completion_stream(self,
                                     messages:List[ChatMessage],
                                     **kwargs,
                                     ) -> AsyncGenerator[str, None]:
        """串流聊天完成"""
        try:
            openai_messages = self._convert_messages(messages)
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as err:
            await self._handle_error(err)
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=False
            )
            return True
        except Exception:
            return False
    
    async def _handle_error(self, error:Exception) -> None:
        """統一錯誤處理"""
        if isinstance(error, RateLimitError):
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(error)}")
        elif isinstance(error, AuthenticationError):
            raise AuthenticationError(f"OpenAI authentication failed: {str(error)}")
        else:
            raise LLMProviderError(f"OpenAI API error: {str(error)}")
