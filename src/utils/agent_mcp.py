"""
MCP (Model Control Protocol) 代理模块
该模块提供了与 MCP 服务器交互的功能，用于处理 AI 模型的查询请求
(使用 OpenAI Tools Agent)
"""

import asyncio
import os
import sys

import uvicorn
from fastapi import FastAPI
from langchain_classic import hub
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils.llm_modle import get_llms

app = FastAPI()

# MCP客户端配置
mcp_config = {
    "howtocook-mcp": {"command": "npx", "args": ["-y", "howtocook-mcp"]},
    # "current_datetime": {
    #     "command": "mcp-proxy",
    #     "args": ["http://127.0.0.1:8001/current_datetime"],
    # },
    # 可以添加更多MCP服务配置
}


async def get_mcp_agent(user_input: str):
    """
    创建并返回一个 MCP 代理实例 (使用 OpenAI Tools Agent)
    Returns:
        AgentExecutor: 配置好的 OpenAI Tools Agent 执行器实例
    """
    async with MultiServerMCPClient(mcp_config) as client:
        model = get_llms(
            supplier="oneapi",
            model="deepseek-ai/DeepSeek-V3",
            api_key=os.getenv("ONEAPI_API_KEY"),  # 填写你的 API Key
        )
        tools = client.get_tools()
        print("--- 可用的工具列表 ---")
        for tool in tools:
            print(f"  名称: {tool.name}")
            print(f"  描述: {tool.description}")
            print(f"  参数 Schema: {tool.args_schema}")
            print("-" * 10)

        # 从 Langchain Hub 拉取适用于 OpenAI Tools Agent 的 Prompt
        prompt = hub.pull("hwchase17/openai-tools-agent")
        # 打印确认 Prompt 输入变量
        print(f"--- 加载的 Prompt 输入变量: {prompt.input_variables} ---")
        # 使用 create_openai_tools_agent 创建 Agent
        agent = create_openai_tools_agent(model, tools, prompt)
        # 创建 Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,  # 工具列表
            verbose=True,  # 是否打印详细日志
        )

        # 调用 invoke 时需要匹配 Prompt 中的输入变量
        # OpenAI Tools Agent Prompt 需要 'input' 和 'chat_history' (作为 MessagesPlaceholder)
        response = await agent_executor.ainvoke(
            {
                "input": user_input,
                "chat_history": [],  # 提供一个空的 chat_history 列表 (或实际历史记录)
            }
        )
        print(f"--- Agent Executor 的最终响应 --- \n{response}")
        return response["output"]


# --- 自定义服务器类 解决Windows上 FastAPI/Asyncio 子进程 `NotImplementedError`
class ProactorServer(uvicorn.Server):
    def run(self, sockets=None):
        # 在服务器运行前设置事件循环策略 (仅 Windows)
        if sys.platform == "win32":
            print("Setting ProactorEventLoopPolicy for Uvicorn server on Windows.")
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        # 使用 asyncio.run 启动服务器的 serve 方法
        asyncio.run(self.serve(sockets=sockets))


if __name__ == "__main__":
    # --- 修改服务器启动方式 ---
    port = 8081
    host = "127.0.0.1"
    print(
        f"Starting MCP Agent server with ProactorServer (OpenAI Tools Agent) at http://{host}:{port}"
    )

    # 1. 创建 Uvicorn 配置，确保 reload=False
    config = uvicorn.Config(app="agent_mcp:app", host=host, port=port, reload=False)

    # 2. 实例化自定义服务器
    server = ProactorServer(config=config)

    # 3. 运行自定义服务器
    server.run()

    # --- 注释掉旧的 uvicorn.run 调用 ---
    # uvicorn.run("mcp_agent:app", host=host, port=port, reload=True)
