# -*- encoding: utf-8 -*-
"""
@File    :  errors.py
@Time    :  2025/08/20 22:00:23
@Author  :  Kevin Wang
@Desc    :  None
"""

from redis import RedisError

class CacheError(Exception):
    """Cache 階段失敗"""
    def __init__(self, msg:str=None):
        super().__init__(msg if msg else "Fail on caching")

class RedisCacheError(CacheError, RedisError):
    """因 Redis 錯誤而造成的 Cache 失敗"""
    def __init__(self, msg:str=None):
        super().__init__(msg if msg else "...")