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
def redis_handler() -> Generator[RedisCacheHandler, None, None]:
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
    def test_connection(self, redis_handler):
        """Test Redis connection"""
        try:
            # 使用 PING 測試連線
            response = redis_handler.redis.ping()
            assert response is True, "Redis connection failed"
        except Exception as err:
            pytest.fail(f"Failed to connect to Redis: {err}")

    def test_build_key(self, redis_handler):
        """Test key building with and without namespace"""
        assert redis_handler._build_key("test_key") == "test_key"
        assert redis_handler._build_key("test_key", "namespace") == "namespace:test_key"

    @pytest.mark.parametrize("key,value", 
                             [("test_string", "test_value"),
                              ("test_float", 1.234),
                              ("test_int", 999),
                              ("test_list", [1, 2, 3]),
                              ("test_complex_obj", datetime.now()),
                              ])
    def test_set_and_get(self, redis_handler, key, value):
        """Test setting and getting various data types"""
        redis_handler.set(key, value)
        result = redis_handler.get(key)
        assert result == value

    def test_set_unpickled_object(self, redis_handler):
        """Test setting and getting complex Python objects"""
        key = "test_dict"
        value = RedisCacheHandler()
        with pytest.raises(TypeError, match="cannot pickle"):
            redis_handler.set(key, value)

    def test_get_nonexistent(self, redis_handler):
        """Test getting a non-existent key"""
        assert redis_handler.get("nonexistent_key") is None

    def test_delete(self, redis_handler):
        """Test deleting a key"""
        key = "test_delete"
        value = "delete_me"
        redis_handler.set(key, value)
        assert redis_handler.get(key) == value
        redis_handler.delete(key)
        assert redis_handler.get(key) is None

    def test_clean(self, redis_handler):
        """Test cleaning all keys"""
        keys = ["key1", "key2", "key3"]
        value = "test_value"
        # Set multiple keys
        for key in keys:
            redis_handler.set(key, value)
        # Verify keys are set
        for key in keys:
            assert redis_handler.get(key) == value
        # Clean all keys
        redis_handler.clean()
        # Verify all keys are removed
        for key in keys:
            assert redis_handler.get(key) is None

    def test_namespace_isolation(self, redis_handler):
        """Test that namespaces properly isolate keys"""
        key = "test_key"
        value1 = "value1"
        value2 = "value2"

        redis_handler.set(key, value1, namespace="ns1")
        redis_handler.set(key, value2, namespace="ns2")

        assert redis_handler.get(key, namespace="ns1") == value1
        assert redis_handler.get(key, namespace="ns2") == value2
