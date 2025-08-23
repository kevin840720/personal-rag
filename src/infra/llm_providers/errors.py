# -*- encoding: utf-8 -*-
"""
@File    :  errors.py
@Time    :  2025/08/23 10:22:03
@Author  :  Kevin Wang
@Desc    :  None
"""

class LLMProviderError(Exception):
    """LLM Provider 基礎錯誤"""
    pass

class RateLimitError(LLMProviderError):
    """速率限制錯誤"""
    pass

class AuthenticationError(LLMProviderError):
    """認證錯誤"""
    pass

class ModelNotFoundError(LLMProviderError):
    """模型不存在錯誤"""
    pass
