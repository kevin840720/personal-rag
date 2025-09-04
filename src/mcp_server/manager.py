# -*- encoding: utf-8 -*-
"""
@File    :  manager.py
@Time    :  2025/08/24 13:39:10
@Author  :  Kevin Wang
@Desc    :  None
"""

import asyncio
from typing import List, Dict, Optional, Iterable, Set

from dotenv import load_dotenv

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

DEBUG_TRACE = True

class LangChainMCPToolServiceManager:
    """
    管理 MCP 工具註冊、啟動以及基於 tags 的權限控制
    """
    def __init__(self):
        self._server_configs:Dict[str, Dict] = {}
        self._server_tags:Dict[str, Set[str]] = {}
        self._tools:Optional[List[BaseTool]] = None
        self._client:Optional[MultiServerMCPClient] = None

    def register_tool(self,
                      name:str,
                      config:Dict,
                      tags:Iterable[str]=None,
                      ):
        """
        Register an MCP server with its configuration and permission tags.
        
        Args:
            name: Unique identifier for the MCP server
            config: Server configuration dictionary
            tags: Optional iterable of permission tags for this server
            
        Raises:
            ValueError: If a server with the same name is already registered
            
        Note:
            The following tags are automatically added:
            - "all": Always present for global access
            - server name: The server name itself is added as a tag
        """
        tags = set(tags) if tags else set()
        tags.add("all")  # 確保 "all" 標籤始終存在
        tags.add(name)   # Server 名稱也視為一種 tag

        if name in self._server_configs:
            raise ValueError(f"MCP Server '{name}' 已經註冊，請使用不同的名稱")

        self._server_configs[name] = config
        self._server_tags[name] = tags

    async def startup(self):
        """
        async 啟動 MCP client
        """
        if self._client is not None:
            raise RuntimeError("MCP client 已啟動")
        self._client = MultiServerMCPClient(self._server_configs)

    async def aget_tools(self,
                         server_name:str,
                         ) -> List[BaseTool]:
        """
        Get tools from a specific MCP server.

        Args:
            server_name: Name of the server to get tools from
            
        Returns:
            List of LangChain BaseTool objects from the specified server

        Note:
            A new session will be created for each tool call.
        """
        if self._client is None:
            raise RuntimeError("Please call startup() first to initialize the MCP client")

        return await self._client.get_tools(server_name=server_name)

    async def aget_tools_by_tags(self,
                                 tags:Iterable[str],
                                 ) -> List[BaseTool]:
        """
        根據 tags 返回對應工具列表（只要工具的 tag 與輸入 tags 有交集即可）
        """
        requested_tags = set(tags)
        
        # 找出所有符合條件的 server names
        matching_servers = [
            name for name, server_tags in self._server_tags.items()
            if server_tags.intersection(requested_tags)
        ]
        
        if not matching_servers:
            return []

        # 並行獲取所有匹配服務器的工具
        tools_collections = await asyncio.gather(*[
            self.aget_tools(name) for name in matching_servers
        ])
        return [tool for tools in tools_collections for tool in tools]
