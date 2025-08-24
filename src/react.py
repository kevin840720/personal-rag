# -*- encoding: utf-8 -*-
"""
@File    :  langchain_react.py
@Time    :  2025/08/23 21:52:37
@Author  :  Kevin Wang
@Desc    :  FIXME: 與 MCP 無關 (可以刪除)，考慮如何把 ReAct 融入 Japanese Learning RAG Tools
            ReACT Agent
            ├── LLM Provider
            ├── Tool Manager
            ├── Memory/Session Manager 
            ├── Action Parser (解析 LLM 輸出的動作)
            ├── Execution Loop (Think->Act->Observe 循環)
            └── Prompt Template (ReACT 格式的提示詞)
"""

import os
import requests
import json
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

class LangChainOpenAIReactAgent:
    def __init__(self,
                 openai_api_key:str=None,
                 mcp_service_url:str="http://localhost:56485",
                 ):
        """初始化 LangChain ReACT Agent"""
        
        # 設置 OpenAI API Key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 初始化 LLM
        self.llm = ChatOpenAI(model="gpt-4.1",
                              temperature=0,
                              streaming=True,
                              )

        # MCP 服務 URL
        self.mcp_service_url = mcp_service_url
        
        # 設置工具
        self.tools = self._setup_tools()
        
        # 載入 ReACT prompt 模板
        self.prompt = hub.pull("hwchase17/react")
        
        # 創建 agent
        self.agent = create_react_agent(llm=self.llm,
                                        tools=self.tools,
                                        prompt=self.prompt,
                                        )
        
        # 創建 agent executor
        self.agent_executor = AgentExecutor(agent=self.agent,
                                            tools=self.tools,
                                            verbose=True,
                                            handle_parsing_errors=True,
                                            max_iterations=10,
                                            )
    
    def _setup_tools(self) -> list:
        """設置可用工具"""
        tools = []
        
        # 1. 搜索工具
        search = DuckDuckGoSearchRun()
        search_tool = Tool(name="Search",
                           func=search.run,
                           description="用於搜索網路資訊。輸入查詢關鍵字，返回相關結果。",
                            )
        # tools.append(search_tool)
        
        # 2. 從 MCP 服務獲取工具
        mcp_tools = self._get_mcp_tools()
        tools.extend(mcp_tools)

        return tools
    
    def _get_mcp_tools(self) -> list:
        """從 MCP 服務獲取工具"""
        tools = []
        
        try:
            # 獲取工具列表
            response = requests.post(
                f"{self.mcp_service_url}/tools",
                json={"tags": ["all"]},
                timeout=10
            )
            
            if response.status_code == 200:
                tools_data = response.json()["tools"]
                
                for tool_info in tools_data:
                    # 為每個 MCP 工具創建包裝器
                    tool_name = tool_info["name"]
                    tool_description = tool_info["description"]
                    
                    def make_tool_wrapper(name):
                        def tool_wrapper(query_input: str) -> str:
                            try:
                                # 解析輸入參數
                                if query_input.strip().startswith('{'):
                                    args = json.loads(query_input)
                                else:
                                    # 根據工具類型構造參數
                                    if "japanese" in name.lower():
                                        args = {"query": query_input}
                                    else:
                                        args = {"input": query_input}
                                
                                # 調用 MCP 服務
                                call_response = requests.post(
                                    f"{self.mcp_service_url}/call",
                                    json={
                                        "tool_name": name,
                                        "args": args
                                    },
                                    timeout=30
                                )
                                
                                if call_response.status_code == 200:
                                    result = call_response.json()
                                    if result["success"]:
                                        return result["result"]
                                    else:
                                        return f"工具調用失敗: {result.get('error', 'Unknown error')}"
                                else:
                                    return f"HTTP 錯誤: {call_response.status_code}"
                                    
                            except Exception as e:
                                return f"工具調用錯誤: {str(e)}"
                        
                        return tool_wrapper
                    
                    langchain_tool = Tool(
                        name=tool_name,
                        func=make_tool_wrapper(tool_name),
                        description=tool_description
                    )
                    tools.append(langchain_tool)
                    
        except Exception as e:
            print(f"無法從 MCP 服務獲取工具: {str(e)}")
        
        return tools
    
    def run(self, query: str) -> str:
        """執行查詢"""
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            return f"執行錯誤：{str(e)}"
    
    def add_tool(self, tool: Tool):
        """添加自定義工具"""
        self.tools.append(tool)
        # 重新創建 agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )


if __name__ == '__main__':
    # 初始化 Agent（會自動從 MCP 服務獲取工具）
    agent = LangChainOpenAIReactAgent(
        openai_api_key=os.getenv("OPEN_AI_API"),
        mcp_service_url="http://localhost:56485"  # MCP 工具服務地址
    )
    
    # 測試查詢
    queries = [
        "你好",
        "日文的助詞有哪些種類？",
        # "什麼是敬語？請解釋用法",
        # "請幫我搜尋一下最新的 AI 新聞",
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"問題: {query}")
        print(f"{'='*50}")
        
        result = agent.run(query)
        print(f"回答: {result}")
