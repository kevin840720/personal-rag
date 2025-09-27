# -*- encoding: utf-8 -*-
"""
測試 monitoring.langfuse_client 的 Langfuse client 取得邏輯。

涵蓋：未提供必要環境變數時應回傳 None，以及以 DummyLangfuse 模擬成功初始化並標記啟用的流程。
"""

import os

import pytest

from conftest import SKIP_LANGFUSE_TESTS
from monitoring.langfuse_client import (REQUIRED_ENV_VARS,
                                        get_langfuse_client,
                                        is_langfuse_enabled,
                                        observe_if_enabled,
                                        )


@pytest.fixture(autouse=True)
def _clear_cache():
    get_langfuse_client.cache_clear()
    yield
    get_langfuse_client.cache_clear()


def test_get_client_returns_none_without_required_env(monkeypatch):
    """確認缺少必要環境變數時返回 None 並判定監控停用。"""
    monkeypatch.setattr("monitoring.langfuse_client.get_client",
                        lambda: object(),
                        raising=False)
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)

    client = get_langfuse_client()

    assert client is None
    assert is_langfuse_enabled() is False


def test_get_client_initializes_when_env_ready(monkeypatch):
    """驗證提供完整設定時可建立 DummyLangfuse 並標記為啟用。"""
    dummy_client = object()

    for key in REQUIRED_ENV_VARS:
        monkeypatch.setenv(key, "value")

    monkeypatch.setattr("monitoring.langfuse_client.get_client",
                        lambda: dummy_client,
                        raising=False)

    client = get_langfuse_client()

    assert client is dummy_client
    assert is_langfuse_enabled() is True


def test_observe_if_disabled_returns_original(monkeypatch):
    """未啟用 Langfuse 時 decorator 應回傳原函式本身。"""
    for key in REQUIRED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("monitoring.langfuse_client.get_client",
                        lambda: None,
                        raising=False)

    def sample(x):
        return x + 1

    decorated = observe_if_enabled(name="demo")(sample)

    assert decorated is sample


def test_observe_if_enabled_applies_decorator(monkeypatch):
    """環境就緒時應呼叫底層 observe decorator。"""
    calls:dict[str,bool] = {"invoked": False}

    for key in REQUIRED_ENV_VARS:
        monkeypatch.setenv(key, "value")

    def fake_observe(**kwargs):
        def wrap(func):
            def inner(*args, **kw):
                calls["invoked"] = True
                return func(*args, **kw)
            return inner
        return wrap

    monkeypatch.setattr("monitoring.langfuse_client.observe", fake_observe, raising=False)
    monkeypatch.setattr("monitoring.langfuse_client.get_client",
                        lambda: object(),
                        raising=False)

    def sample(x):
        return x + 1

    decorated = observe_if_enabled(name="demo")(sample)

    assert decorated(1) == 2
    assert calls["invoked"] is True


@pytest.mark.skipif(SKIP_LANGFUSE_TESTS, reason="Skipping LangFuse tests")
def test_langfuse_enabled_with_real_env():
    """若環境已配置且安裝 Langfuse，應能啟用實際 client。"""
    from dotenv import load_dotenv

    load_dotenv()

    from monitoring import langfuse_client as module

    if module.get_client is None:
        pytest.fail("langfuse 套件未安裝，請安裝後再執行實測")

    missing = [key for key in module.REQUIRED_ENV_VARS if not os.getenv(key)]
    if missing:
        pytest.fail(f"缺少必要環境變數: {', '.join(missing)}")

    client = get_langfuse_client()

    assert client is not None
    assert is_langfuse_enabled() is True
