# -*- encoding: utf-8 -*-
"""
@File    :  rag_tools.py
@Time    :  2025/08/23 19:30:01
@Author  :  Kevin Wang
@Desc    :  None
"""

from dotenv import load_dotenv

from mcp_server.jp_learning_rag import JapaneseLearningRAGServer

load_dotenv()

if __name__ == "__main__":
    server = JapaneseLearningRAGServer(host="localhost", port=56481)
    server.run(transport="streamable-http")