# -*- encoding: utf-8 -*-
"""
@File    :  base.py
@Time    :  2025/08/20 22:00:04
@Author  :  Kevin Wang
@Desc    :  None
"""


from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any
import hashlib

class BaseCacheHandler(ABC):
    """Cache handling base class for embedding vectors"""
    def __init__(self,
                 cache_ttl:int=24*60*60,
                 ):
        """Initialize cache handler
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default 24 hours)
        """
        self.cache_ttl = cache_ttl

    def get_cache_key(self,
                      text:str,
                      ) -> str:
        """Generate cache key for input text

        Args:
            text: Input text to generate key for
            
        Returns:
            Cache key string
        """
        return hashlib.md5(text.encode()).hexdigest()

    @abstractmethod
    def get(self,
            key:str,
            ) -> Optional[Any]:
        """Retrieve value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and valid, None otherwise
        """
        pass

    @abstractmethod
    def set(self,
            key:str,
            value:Any,
            ) -> None:
        """Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        pass

    @abstractmethod
    def delete(self,
               key:str,
               ) -> None:
        """Delete key from cache
        
        Args:
            key: Cache key to delete
        """
        pass
