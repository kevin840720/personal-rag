# -*- encoding: utf-8 -*-
"""
@File    :  mcp_service_server.py
@Time    :  2025/08/24 13:45:00
@Author  :  Kevin Wang
@Desc    :  MCP 工具服務器，監聽 port 56485
"""

import asyncio
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from mcp_server.manager import LangChainMCPToolServiceManager

load_dotenv()

# 日誌設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Tools Service", version="1.0.0")

# 全局管理器
manager = None

class ToolsRequest(BaseModel):
    tags: List[str]

class ToolsResponse(BaseModel):
    tools: List[Dict[str, Any]]

class ToolCallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any]

class ToolCallResponse(BaseModel):
    result: str
    success: bool
    error: str = None

@app.on_event("startup")
async def startup_event():
    """啟動時初始化 MCP 管理器"""
    global manager
    try:
        logger.info("初始化 MCP 管理器...")
        manager = LangChainMCPToolServiceManager()
        
        # 註冊工具
        manager.register_tool("japanese-learning-note",
                            {"command": "env",
                             "args": ["PYTHONPATH=src", "pipenv", "run", "python", "src/mcp_server/jp_learning_rag.py"],
                             "transport": "stdio",
                             },
                            tags=["all", "japanese-learning"],
                            )
        
        # manager.register_tool("sequential-thinking",
        #                     {"command": "docker",
        #                      "args": ["run", "--rm", "-i", 
        #                               "mcp/sequentialthinking",
        #                               ],
        #                      "transport": "stdio",
        #                      },
        #                     tags=["all", "general"],
        #                     )
        
        # 啟動管理器
        await manager.startup()
        logger.info("MCP 管理器初始化完成")
        
    except Exception as e:
        logger.error(f"MCP 管理器初始化失敗: {str(e)}")
        raise

@app.post("/tools", response_model=ToolsResponse)
async def get_tools(request: ToolsRequest):
    """根據標籤獲取工具列表"""
    global manager
    
    if not manager:
        raise HTTPException(status_code=500, detail="Manager not initialized")
    
    try:
        tools = await manager.aget_tools_by_tags(request.tags)
        
        # 轉換為可序列化的格式
        tools_data = []
        for tool in tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args if hasattr(tool, 'args') else {}
            })
        
        return ToolsResponse(tools=tools_data)
        
    except Exception as e:
        logger.error(f"獲取工具失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """調用指定工具"""
    global manager
    
    if not manager:
        raise HTTPException(status_code=500, detail="Manager not initialized")
    
    try:
        # 獲取所有工具
        all_tools = await manager.aget_tools_by_tags(["all"])
        
        # 找到對應的工具
        target_tool = None
        for tool in all_tools:
            if tool.name == request.tool_name:
                target_tool = tool
                break
        
        if not target_tool:
            return ToolCallResponse(
                result="",
                success=False,
                error=f"Tool '{request.tool_name}' not found"
            )
        
        # 調用工具
        result = await target_tool.ainvoke(request.args)
        
        return ToolCallResponse(
            result=str(result),
            success=True
        )
        
    except Exception as e:
        logger.error(f"工具調用失敗: {str(e)}")
        return ToolCallResponse(
            result="",
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy" if manager else "unhealthy",
        "tools_count": len(await manager.aget_tools_by_tags(["all"])) if manager else 0
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost", 
        port=56485,
        log_level="info"
    )