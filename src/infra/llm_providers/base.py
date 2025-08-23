# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/08/23 10:12:49
@Author  :  Kevin Wang
@Desc    :  None
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role:str
    content:str
    metadata:Optional[Dict[str,Any]] = None

class LLMConfig(BaseModel):
    model_name:str
    temperature:float=0.7
    max_tokens:Optional[int]=None
    streaming:bool=False

class AbstractLLMProvider(ABC):
    def __init__(self, config:LLMConfig):
        self.config = config
    
    @abstractmethod
    async def chat_completion(self, 
                              messages:List[ChatMessage],
                              **kwargs,
                              ) -> ChatMessage:
        """非串流聊天完成"""
        pass
    
    @abstractmethod
    async def chat_completion_stream(self,
                                     messages:List[ChatMessage],
                                     **kwargs,
                                     ) -> AsyncGenerator[str, None]:
        """串流聊天完成"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康檢查"""
        pass