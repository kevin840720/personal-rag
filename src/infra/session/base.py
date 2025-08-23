# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/08/23 15:33:20
@Author  :  Kevin Wang
@Desc    :  None
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel

from infra.llm_providers.base import ChatMessage

class ConversationSession(BaseModel):
    session_id:str
    messages:List[ChatMessage]
    context:Dict[str,Any]
    created_at:datetime
    updated_at:datetime

class AbstractSessionManager(ABC):
    @abstractmethod
    async def create_session(self,
                             session_id:Optional[str]=None,
                             ) -> str:
        """建立新會話"""
        pass
    
    @abstractmethod
    async def get_session(self,
                          session_id:str,
                          ) -> Optional[ConversationSession]:
        """取得會話"""
        pass
    
    @abstractmethod
    async def add_message(self,
                          session_id:str,
                          message:ChatMessage,
                          ) -> None:
        """新增訊息到會話"""
        pass
    
    @abstractmethod
    async def delete_session(self,
                             session_id:str,
                             ) -> bool:
        """刪除會話"""
        pass
