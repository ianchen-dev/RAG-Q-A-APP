import logging
import os
from typing import Any, AsyncGenerator, Dict, List

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# BaseMessage 导入已移除，因为 astream_events 不需要直接处理消息类型
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from src.components.llm_provider import get_llms

# 从管理器导入MCP工具获取函数
from src.config.mcp_client_manager import get_cached_mcp_tools

# Import tools from the new src.tools package
from src.tools import (
    create_tavily_tool,
    get_knowledge_list_tool,
    retriever_document_tool,
)

logger = logging.getLogger(__name__)


# 移除 StreamingCallbackHandler，改用 RunnableWithMessageHistory.astream 方法


class LangChainAgent:
    """基于 LangChain 的 Agent 实现，功能与 LangGraph 版本保持一致"""

    def __init__(self):
        # MongoDB 配置，保持与 ChatSev.py 一致
        self.mongo_connection_string = os.getenv("MONGODB_URL")
        if not self.mongo_connection_string:
            raise ValueError(
                "错误：环境变量 MONGODB_URL 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )
        self.mongo_database_name = os.getenv("MONGO_DB_NAME")
        if not self.mongo_database_name:
            raise ValueError(
                "错误：环境变量 MONGO_DB_NAME 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )
        self.mongo_collection_name = os.getenv("MONGODB_COLLECTION_NAME_CHATHISTORY")
        if not self.mongo_collection_name:
            raise ValueError(
                "错误：环境变量 MONGODB_COLLECTION_NAME_CHATHISTORY 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )

        # 基础工具配置 - 条件性添加 Tavily 搜索工具
        self.base_tools = [get_knowledge_list_tool, retriever_document_tool]

        # 如果 TavilySearch 可用，则添加到工具列表
        tavily_tool = create_tavily_tool(max_results=2)
        if tavily_tool is not None:
            self.base_tools.append(tavily_tool)

        # Tool Calling Agent 提示模板 - 支持原生工具调用
        self.tool_calling_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个助手，你擅长使用工具帮助用户解决问题。你拥有知识库检索工具（查询本地知识库，检索其中的内容，回答用户问题）、联网搜索的工具、食谱推荐工具、高德地图工具（路线规划等）。你会先告知用户你将用xx工具来解决问题，再调用工具，获取需要的信息后，给用户答复",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

    def get_session_chat_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """根据 session_id 获取 MongoDB 聊天记录实例，与 ChatSev.py 保持一致"""
        logger.info(f"获取 session_id 为 {session_id} 的 MongoDB 聊天记录")
        return MongoDBChatMessageHistory(
            connection_string=self.mongo_connection_string,
            session_id=session_id,
            database_name=self.mongo_database_name,
            collection_name=self.mongo_collection_name,
        )

    async def _get_all_tools(self) -> List[Any]:
        """获取所有可用工具（基础工具 + MCP 工具）"""
        # 从管理器获取预加载的 MCP 工具
        mcp_tools = await get_cached_mcp_tools()
        if not mcp_tools:
            logger.warning(
                "MCP tools not available from manager. Proceeding with base tools only."
            )
        else:
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools from manager.")
            for m_tool in mcp_tools:
                logger.info(f"  MCP Tool: {m_tool.name}")

        all_tools = self.base_tools + mcp_tools
        logger.info("--- All Tools for LangChain Agent ---")
        for tool in all_tools:
            logger.info(f"  Name: {tool.name}")
        logger.info("-" * 40)

        return all_tools

    # _create_agent_executor 方法已移除，因为 stream_chat 中直接创建 AgentExecutor

    async def stream_chat(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户输入，使用 astream_events 方法实现真正的 token 级流式输出

        Args:
            user_input: 用户输入的问题
            session_id: 会话ID，用于历史记录管理

        Yields:
            Dict[str, Any]: 包含 'type' 和 'data' 的事件字典
            - {"type": "chunk", "data": "token"}  # 真正的 token 级流式输出
            - {"type": "tool_call", "data": {...}}  # 工具调用事件
            - {"type": "tool_result", "data": {...}}  # 工具结果事件
            - {"type": "stream_end"}  # 流结束标识
        """
        try:
            logger.info(f"开始处理用户输入: '{user_input}' (session_id: {session_id})")

            # 获取 LLM
            # 硅基流动出现工具参数不兼容，故在agent模式下弃用
            # llm = get_llms(
            #     supplier="siliconflow",
            #     model="deepseek-ai/DeepSeek-V3",
            #     api_key=os.getenv("SILICONFLOW_API_KEY"),
            #     streaming=True,  # 显式启用流式输出
            # )

            llm = get_llms(supplier="volces", model="deepseek")

            # 验证 LLM 是否支持流式输出
            logger.info(f"LLM 流式输出配置: {getattr(llm, 'streaming', 'unknown')}")

            # 获取所有工具
            all_tools = await self._get_all_tools()

            # 创建 Tool Calling Agent
            agent = create_tool_calling_agent(llm, all_tools, self.tool_calling_prompt)

            # 创建 AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True,
                streaming=True,
            )
            logger.info("AgentExecutor 创建成功")

            # 包装历史记录管理 - 模仿 ChatSev.py 的做法
            agent_with_history = RunnableWithMessageHistory(
                agent_executor,
                self.get_session_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="output",  # AgentExecutor 的输出键
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="session_id",
                        annotation=str,
                        name="Session ID",
                        description="Unique identifier for the chat session.",
                        default="",
                        is_shared=True,
                    )
                ],
            )

            # 准备配置和输入
            config = {"configurable": {"session_id": session_id}}
            agent_input = {"input": user_input}

            logger.info(f"开始流式执行 agent (session_id: {session_id})")

            # 用于收集完整输出以保存到历史记录
            collected_output = []

            # 使用 astream_events 进行真正的 token 级流式处理
            # 尝试不同的版本参数或不使用版本参数
            async for event in agent_with_history.astream_events(
                agent_input,
                config=config,
                # 暂时移除 version 参数，看看是否有区别
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # 详细日志记录所有事件类型，用于调试
                # logger.info(f"事件类型: {event_type}, 名称: {event_name}, 数据: {event_data}")

                # 处理所有可能的流式事件类型

                # 处理 LLM token 流式输出 - 这是真正的 token 级流式输出！
                if event_type == "on_llm_stream":
                    logger.info(f"LLM 流事件: {event_data}")
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content
                        if token:  # 发送真正的 token 级流式输出
                            collected_output.append(token)
                            logger.info(f"发送 token: '{token}'")
                            yield {"type": "chunk", "data": token}

                # 处理 Chat Model 流式输出（另一种可能的事件类型）
                elif event_type == "on_chat_model_stream":
                    logger.info(f"Chat Model 流事件: {event_data}")
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content
                        if token:
                            collected_output.append(token)
                            logger.info(f"发送 token (chat_model): '{token}'")
                            yield {"type": "chunk", "data": token}

                # 处理工具调用开始事件
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get("input", {})
                    # 使用事件的运行 ID 确保唯一性和一致性
                    run_id = event.get("run_id", "")
                    call_id = (
                        f"call_{run_id}_{tool_name}"
                        if run_id
                        else f"call_{hash(str(tool_name) + str(tool_input))}"
                    )

                    # 存储 call_id 映射，用于后续的 tool_result 事件
                    if not hasattr(self, "_tool_call_mapping"):
                        self._tool_call_mapping = {}
                    # 使用 run_id 作为键，确保匹配
                    mapping_key = f"{run_id}_{tool_name}" if run_id else tool_name
                    self._tool_call_mapping[mapping_key] = call_id

                    logger.info(f"生成工具调用事件: {tool_name}, call_id: {call_id}")
                    yield {
                        "type": "tool_call",
                        "data": {"name": tool_name, "args": tool_input, "id": call_id},
                    }

                # 处理工具调用结束事件
                elif event_type == "on_tool_end":
                    tool_name = event_name
                    tool_output = event_data.get("output")
                    run_id = event.get("run_id", "")

                    # 使用相同的映射键获取 call_id
                    mapping_key = f"{run_id}_{tool_name}" if run_id else tool_name
                    call_id = getattr(self, "_tool_call_mapping", {}).get(
                        mapping_key, f"call_{hash(str(tool_name))}"
                    )

                    logger.info(
                        f"生成工具结果事件: {tool_name}, tool_call_id: {call_id}"
                    )
                    yield {
                        "type": "tool_result",
                        "data": {
                            "name": tool_name,
                            "content": str(tool_output),
                            "tool_call_id": call_id,
                        },
                    }

                # 处理 Agent 动作事件
                elif event_type == "on_agent_action":
                    action = event_data.get("action")
                    if action:
                        yield {
                            "type": "tool_call",
                            "data": {
                                "name": action.tool,
                                "args": action.tool_input,
                                "id": f"call_{hash(str(action))}",
                            },
                        }

                # 处理 Chain 流式输出（备用方案）
                elif event_type == "on_chain_stream":
                    logger.info(f"Chain 流事件: {event_data}")
                    chunk = event_data.get("chunk")
                    if isinstance(chunk, dict) and "output" in chunk:
                        output_content = chunk["output"]
                        if isinstance(output_content, str) and output_content.strip():
                            # 如果没有通过 LLM 流捕获到 token，使用这个作为备用
                            if not collected_output:
                                collected_output.append(output_content)
                                logger.info(f"发送 chain 输出: '{output_content}'")
                                yield {"type": "chunk", "data": output_content}

                # 通用事件处理器 - 捕获所有包含 chunk 的流式事件
                elif "stream" in event_type and event_data:
                    logger.info(f"通用流事件 {event_type}: {event_data}")
                    chunk = event_data.get("chunk")
                    if chunk:
                        # 尝试不同的 chunk 结构
                        content = None
                        if hasattr(chunk, "content"):
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        elif isinstance(chunk, dict) and "content" in chunk:
                            content = chunk["content"]

                        if content and content.strip():
                            collected_output.append(content)
                            logger.info(f"发送通用流内容: '{content}'")
                            yield {"type": "chunk", "data": content}

            # 发送流结束信号
            yield {"type": "stream_end"}
            logger.info(f"Agent 流式处理完成 (session_id: {session_id})")

        except Exception as e:
            logger.error(
                f"流式处理时发生错误 (session_id: {session_id}): {e}", exc_info=True
            )
            yield {"type": "error", "data": f"处理请求时发生错误: {e}"}


# 全局实例变量（懒加载）
_langchain_agent_instance = None


def get_langchain_agent():
    """获取 LangChain Agent 实例（懒加载模式）"""
    global _langchain_agent_instance
    if _langchain_agent_instance is None:
        _langchain_agent_instance = LangChainAgent()
    return _langchain_agent_instance


async def main_graph_execution(
    user_input: str,
    session_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    主要执行函数，保持与原 react_agent.py 相同的接口

    Args:
        user_input: 用户输入
        session_id: 会话ID

    Yields:
        Dict[str, Any]: 事件流字典，包含以下类型：
        - {"type": "tool_call", "data": {...}}    # 工具调用开始
        - {"type": "tool_result", "data": {...}}  # 工具执行结果
        - {"type": "chunk", "data": "..."}        # 流式文本输出
        - {"type": "stream_end"}                  # 流结束标识
        - {"type": "error", "data": "..."}        # 错误信息
    """
    langchain_agent = get_langchain_agent()
    async for event in langchain_agent.stream_chat(user_input, session_id):
        yield event


# Integration tests have been migrated to test/integration/service/test_langchain_agent.py
# Run with: uv run pytest test/integration/service/test_langchain_agent.py -v
