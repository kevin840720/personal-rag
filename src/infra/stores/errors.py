# -*- encoding: utf-8 -*-
"""
@File    :  errors.py
@Time    :  2025/01/20 13:59:12
@Author  :  Kevin Wang
@Desc    :  None
"""

class IndexStoreError(Exception):
    """Index Store 失敗"""
    def __init__(self, msg:str=None):
        super().__init__(msg if msg else "Fail on embedding")

class ChromaActionError(IndexStoreError):
    """ChromaDB 操作失敗"""
    # 由於 ChromaDB 的套件時常有「不報錯」的問題，所以部份情形需要手動檢查（如：ID Conflict、Update non-existed ID、etc）
    # 此類別的作用是為了那些「手動檢查」的項目報錯
    def __init__(self, msg:str=None):
        super().__init__(msg if msg else "ChromaDB 操作過程中發生錯誤")

class ElasticsearchActionError(IndexStoreError):
    """Elasticsearch 操作失敗"""
    def __init__(self, msg:str=None):
        super().__init__(msg if msg else "Elasticsearch 操作過程中發生錯誤")
