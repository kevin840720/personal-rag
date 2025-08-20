# -*- encoding: utf-8 -*-
"""
@File    :  test_redis.py
@Time    :  2025/08/20 22:02:09
@Author  :  Kevin Wang
@Desc    :  None
"""

import os
from datetime import datetime
from typing import Generator

import pytest
from dotenv import load_dotenv

from cache.redis import RedisCacheHandler
from conftest import SKIP_REDIS_TESTS

load_dotenv()

@pytest.fixture
def store() -> Generator[RedisCacheHandler, None, None]:
    """Fixture to create a Redis cache handler instance"""
    handler = RedisCacheHandler(host=os.getenv("MY_REDIS_HOST"),
                                port=os.getenv("MY_REDIS_PORT"),
                                password=os.getenv("MY_REDIS_PASSWORD"),
                                default_cache_ttl=300,
                                )
    yield handler
    # Clean up after each test
    handler.clean()

@pytest.mark.skipif(SKIP_REDIS_TESTS, reason="Skipping Redis test")
class TestRedisCacheHandler:
    def test_connection(self, store:RedisCacheHandler):
        """Test Redis connection"""
        try:
            # 使用 PING 測試連線
            response = store.redis.ping()
            assert response is True, "Redis connection failed"
        except Exception as err:
            pytest.fail(f"Failed to connect to Redis: {err}")

    def test_build_key(self, store:RedisCacheHandler):
        """Test key building with and without namespace"""
        assert store._build_key("test_key") == "test_key"
        assert store._build_key("test_key", "namespace") == "namespace:test_key"

    @pytest.mark.parametrize("key,value", 
                             [("test_string", "test_value"),
                              ("test_float", 1.234),
                              ("test_int", 999),
                              ("test_list", [1, 2, 3]),
                              ("test_complex_obj", datetime.now()),
                              ])
    def test_set_and_get(self,
                         store:RedisCacheHandler,
                         key:str,
                         value,  # pickle-able type
                         ):
        """Test setting and getting various data types"""
        store.set(key, value)
        result = store.get(key)
        assert result == value

    def test_set_unpickled_object(self, store:RedisCacheHandler):
        """Test setting and getting complex Python objects"""
        key = "test_dict"
        value = RedisCacheHandler()
        with pytest.raises(TypeError, match="cannot pickle"):
            store.set(key, value)

    def test_get_nonexistent(self, store:RedisCacheHandler):
        """Test getting a non-existent key"""
        assert store.get("nonexistent_key") is None

    def test_delete(self, store:RedisCacheHandler):
        """Test deleting a key"""
        key = "test_delete"
        value = "delete_me"
        store.set(key, value)
        assert store.get(key) == value
        store.delete(key)
        assert store.get(key) is None

    def test_clean(self, store:RedisCacheHandler):
        """Test cleaning all keys"""
        keys = ["key1", "key2", "key3"]
        value = "test_value"
        # Set multiple keys
        for key in keys:
            store.set(key, value)
        # Verify keys are set
        for key in keys:
            assert store.get(key) == value
        # Clean all keys
        store.clean()
        # Verify all keys are removed
        for key in keys:
            assert store.get(key) is None

    def test_namespace_isolation(self, store:RedisCacheHandler):
        """Test that namespaces properly isolate keys"""
        key = "test_key"
        value1 = "value1"
        value2 = "value2"

        store.set(key, value1, namespace="ns1")
        store.set(key, value2, namespace="ns2")

        assert store.get(key, namespace="ns1") == value1
        assert store.get(key, namespace="ns2") == value2
