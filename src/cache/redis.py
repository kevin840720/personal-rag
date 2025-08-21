# -*- encoding: utf-8 -*-
"""
@File      : redis.py
@Time      : 2025/01/16 18:12:54
@Author    : Kevin Wang
@Desc      :
@Notes     : 
    1. Redis 支援多個資料庫，透過 DB number 區分（僅支援數字標識）。如需隔離不同應用，應指定不同的 DB number。
    2. 鍵名中的冒號 `:` 是 Redis 的通用約定，用於表示命名空間（Namespace），例如 `ns:key`。
    3. 預設採用二進制存儲（`decode_responses=False`），所有數據均會經過 pickle 序列化。
    4. 批量操作（如 `batch_set` 和 `batch_delete`）需手動管理命名空間。
    5. TTL（存活時間）默認為 1 週，可透過 `default_cache_ttl` 配置。
@DevNotes  :
    為何使用 pickle 將資料序列化存儲：
    1. 支援儲存更多型別的資料（如 dict, list, 自定義物件等）。
    2. 解決 Redis 內部將數據（如 int/float）自動轉為字串的問題，確保還原後能保持原始型別。
        - 例如：儲存 999（int）與 "999"（str）到 Redis 中，取出時都會變為字串形式 "999"，導致型別無法辨識。
    3. 缺點：使用 pickle 後無法透過相關 GUI 界面直接看到資料內容。
"""

from typing import (Any,
                    List,
                    Optional,
                    )
import pickle

import redis

from cache.base import BaseCacheHandler
from cache.errors import (CacheError,
                          RedisCacheError,
                          )

class RedisCacheHandler(BaseCacheHandler):
    """Redis-based cache implementation"""
    def __init__(self,
                 host:str="localhost",
                 port:int=36379,
                 password:Optional[str]=None,
                 **kwargs,
                 ):
        """Initialize Redis cache handler.
        
        Args:
            host (str): Redis server host. Defaults to "localhost".
            port (int): Redis server port. Defaults to 36379.
            password (Optional[str]): Redis server password. Defaults to None.
            default_cache_ttl (int, optional): Cache time-to-live in seconds. Defaults to 1 week.
        """
        self._default_cache_ttl = kwargs.pop("default_cache_ttl", 31*24*60*60)
        self.redis = redis.Redis(host=host,
                                 port=port,
                                 password=password,
                                 decode_responses=False,  # Keep binary data for pickle
                                 **kwargs,
                                 )

    def _build_key(self,
                   key:str,
                   namespace:Optional[str]="",
                   ) -> str:
        """Construct a namespaced key.
        
        Args:
            key (str): The original key.
            namespace (Optional[str]): Namespace to prefix the key. Defaults to "".
        
        Returns:
            str: The namespaced key.
        """
        return f"{namespace}:{key}" if namespace else key

    def get(self,
            key:str,
            namespace:Optional[str]="",
            ) -> Optional[Any]:
        """Retrieve a value from Redis cache.
        
        Args:
            key (str): The cache key to retrieve.
            namespace (Optional[str]): Namespace to distinguish keys. Defaults to "".
        
        Returns:
            Optional[Any]: The deserialized value if it exists, otherwise None.
        
        Raises:
            CacheError: If deserialization fails.
            RedisCacheError: If a Redis-related error occurs.
        """
        try:
            namespaced_key = self._build_key(key, namespace)
            data = self.redis.get(namespaced_key)
            if data is None:
                return None
            return pickle.loads(data)
        except (pickle.PickleError, TypeError) as err:
            raise CacheError(f"Failed to deserialize cache for key {key} in namespace {namespace}: {err}") from err
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while retrieving key {key} in namespace {namespace}: {err}") from err

    def set(self,
            key:str,
            value:Any,
            namespace:Optional[str]="",
            ttl:Optional[int]=None,
            ) -> None:
        """
        Store a value in Redis cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to store. It will be serialized using pickle.
            namespace (Optional[str]): Namespace to distinguish keys. Defaults to "".
            ttl (Optional[int]): Time-to-live in seconds. Defaults to default_cache_ttl.
        
        Raises:
            RedisCacheError: If a Redis-related error occurs.
        """
        try:
            namespaced_key = self._build_key(key, namespace)
            # 由於 Redis 是二進制儲存，他會將所有
            value = pickle.dumps(value)
            self.redis.setex(name=namespaced_key,
                             time=ttl or self._default_cache_ttl,
                             value=value,
                             )
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while setting key {key} in namespace {namespace}: {err}") from err

    def batch_set(self,
                  items:dict,
                  ttl:Optional[int]=None) -> None:
        """Batch store multiple key-value pairs in Redis.
        
        注意：
            如須使用命名空間 (namespace)，需要自行構造鍵名。例如：
            - 單次設定：set("key", "val", "ns")
            - 批量設定等效於：batch_set({"ns:key": "val"})

        Args:
            items (dict): 鍵值對字典，其中鍵為 Redis 鍵（包含 namespace，如需要），值為對應的值。
            ttl (Optional[int]): 所有鍵的存活時間（以秒為單位）。默認為 default_cache_ttl。
        
        Raises:
            RedisCacheError: 如果執行批量操作時發生 Redis 錯誤。
        """
        try:
            with self.redis.pipeline() as pipe:
                for key, value in items.items():
                    value = pickle.dumps(value)
                    pipe.setex(name=key,
                               time=ttl or self._default_cache_ttl,
                               value=value,
                               )
                pipe.execute()
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while performing batch set: {err}") from err

    def delete(self,
               key:str,
               namespace:Optional[str] = "",
               ) -> None:
        """Delete a key from Redis cache.
        
        Args:
            key (str): The cache key to delete.
            namespace (Optional[str]): Namespace to distinguish keys. Defaults to "".
        
        Raises:
            RedisCacheError: If a Redis-related error occurs.
        """
        try:
            namespaced_key = self._build_key(key, namespace)
            self.redis.delete(namespaced_key)
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while deleting key {key} in namespace {namespace}: {err}") from err

    def batch_delete(self,
                     keys:List[str],
                     ) -> None:
        """Batch delete multiple keys from Redis.
        
        注意：
            如須使用命名空間 (namespace)，需要自行構造鍵名。例如：
            - 單次設定：delete("key", "ns")
            - 批量設定等效於：batch_delete(["ns:key"])

        Args:
            keys (list): A list of keys to delete.
            namespace (Optional[str]): Namespace to distinguish keys.
        """
        try:
            with self.redis.pipeline() as pipe:
                for key in keys:
                    pipe.delete(key)
                pipe.execute()
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while performing batch delete: {err}") from err

    def clean(self) -> None:
        """Delete all keys from the current Redis database.
        
        Raises:
            RedisCacheError: If a Redis-related error occurs.
        """
        try:
            self.redis.flushdb()
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while cleaning the database: {err}") from err

    def reset_ttl(self,
                  new_ttl:Optional[int]=None,
                  batch_size:int=1000,
                  ) -> None:
        """Reset the TTL (time-to-live) for all keys in Redis.
        
        Args:
            new_ttl (Optional[int]): The new TTL in seconds. Defaults to default_cache_ttl.
            batch_size (int): The number of keys to process per pipeline batch. Defaults to 1000.
        
        Raises:
            RedisCacheError: If a Redis-related error occurs.
        """
        new_ttl = new_ttl or self._default_cache_ttl
        try:
            cursor = '0'  # 由於使用 Scan，所以要用 Cursor 來紀錄當前位置
            pipeline = self.redis.pipeline()
            commands_in_pipeline = 0
            while cursor != 0:
                cursor, keys = self.redis.scan(cursor=cursor, match='*', count=batch_size)
                for key in keys:
                    pipeline.expire(key, new_ttl)
                    commands_in_pipeline += 1

                    # 當管道中的命令達到 batch_size 時，執行並重置管道
                    if commands_in_pipeline >= batch_size:
                        pipeline.execute()
                        pipeline = self.redis.pipeline()
                        commands_in_pipeline = 0
            # 執行剩餘的命令
            if commands_in_pipeline > 0:
                pipeline.execute()
        except redis.RedisError as err:
            raise RedisCacheError(f"Redis error occurred while recalculating TTL: {err}") from err
