# -*- encoding: utf-8 -*-
"""Langfuse Python SDK (v3) helper utilities."""

import logging
import os
from functools import lru_cache, wraps
from typing import Any, Callable, Optional, TypeVar, cast

from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency
    from langfuse import get_client, observe
except ImportError:  # pragma: no cover - tests without langfuse installed
    get_client = None  # type: ignore
    observe = None  # type: ignore

try:  # pragma: no cover - context helper is optional
    from langfuse import langfuse_context  # type: ignore
except ImportError:  # pragma: no cover
    langfuse_context = None  # type: ignore

load_dotenv()

LOGGER = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ("LANGFUSE_HOST",
                     "LANGFUSE_PUBLIC_KEY",
                     "LANGFUSE_SECRET_KEY",
                     )

F = TypeVar("F", bound=Callable[..., Any])


def _env_ready() -> bool:
    return all(os.getenv(k) for k in REQUIRED_ENV_VARS)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Optional[Any]:
    """Lazy-initialize Langfuse client when環境設定齊備。"""
    if get_client is None:
        LOGGER.warning("langfuse 套件未安裝，將略過 Langfuse 監控初始化")
        return None
    if not _env_ready():
        LOGGER.info("缺少 Langfuse 連線設定，監控功能停用")
        return None
    try:
        return get_client()
    except Exception as exc:  # pragma: no cover - network failures
        LOGGER.warning("初始化 Langfuse client 失敗：%s", exc)
        return None


def is_langfuse_enabled() -> bool:
    return get_langfuse_client() is not None


def observe_if_enabled(*,
                       name: Optional[str] = None,
                       as_type: Optional[str] = None,
                       capture_input: bool = True,
                       capture_output: bool = True,
                       ) -> Callable[[F], F]:
    """Wrap Langfuse @observe，未啟用時回傳原函式。"""

    def decorator(func: F) -> F:
        if observe is None or not is_langfuse_enabled():
            return func
        wrapped = observe(name=name,
                          as_type=cast("Optional[str]", as_type),
                          capture_input=capture_input,
                          capture_output=capture_output,
                          )(func)
        return cast(F, wrapped)

    return decorator


__all__ = ["REQUIRED_ENV_VARS",
           "get_langfuse_client",
           "is_langfuse_enabled",
           "observe_if_enabled",
           "langfuse_context",
           ]
