import logging
import os
from typing import Any, AsyncGenerator, Dict, List

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
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


class LangChainReActAgent:
    """基于 LangChain 的 ReAct Agent 实现，支持推理-行动模式"""

    def __init__(self):
        # MongoDB 配置，与原版本保持一致
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

        # ReAct Agent 提示模板 - 使用自定义模板以支持聊天历史
        # 注意：标准的 hwchase17/react 模板不支持 chat_history，所以我们使用自定义模板

        # 自定义 ReAct 提示模板 - 包含必需的变量: tools, tool_names, agent_scratchpad, input
        # 同时支持可选的 chat_history 变量
        self.react_prompt = PromptTemplate.from_template("""
你是一个助手，擅长使用工具帮助用户解决问题。你拥有以下工具：

{tools}

请按照以下格式进行推理和行动：

Question: 用户的问题
Thought: 我需要思考如何解决这个问题
Action: 我选择使用的工具名称，应该是以下之一 [{tool_names}]
Action Input: 工具的输入参数（如果工具需要复杂参数，请使用 JSON 格式，例如: {{"key": "value"}}）
Observation: 工具执行的结果
... (这个思考/行动/观察的过程可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

重要提示：
- 对于需要多个参数的工具，请将 Action Input 格式化为 JSON，例如: {{"city": "北京", "type": "weather"}}
- 对于简单的单个参数工具，可以直接使用字符串
- 请仔细查看每个工具的描述，了解它需要什么格式的输入

开始！

Question: {input}
Thought:{agent_scratchpad}
""")

        logger.info("使用自定义 ReAct 提示模板（支持聊天历史）")

    def get_session_chat_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """根据 session_id 获取 MongoDB 聊天记录实例，与原版本保持一致"""
        logger.info(f"获取 session_id 为 {session_id} 的 MongoDB 聊天记录")
        return MongoDBChatMessageHistory(
            connection_string=self.mongo_connection_string,
            session_id=session_id,
            database_name=self.mongo_database_name,
            collection_name=self.mongo_collection_name,
        )

    def _create_react_compatible_tool_wrapper(self, original_tool):
        """为 ReAct Agent 创建兼容的工具包装器，处理字符串输入问题"""
        import json

        from langchain_core.tools import Tool

        async def wrapped_tool_func(tool_input: str) -> str:
            """包装器函数：将字符串输入转换为工具期望的格式"""
            logger.info(
                f"工具 {original_tool.name} 收到输入: '{tool_input}' (类型: {type(tool_input)})"
            )

            try:
                # 如果工具有 args_schema，优先尝试解析 JSON 输入
                if hasattr(original_tool, "args_schema") and original_tool.args_schema:
                    logger.info(
                        f"工具 {original_tool.name} 有 args_schema，尝试 JSON 解析"
                    )
                    try:
                        # 尝试将字符串解析为 JSON
                        parsed_input = json.loads(tool_input)
                        logger.info(f"JSON 解析成功: {parsed_input}")
                        result = await original_tool.ainvoke(parsed_input)
                    except (json.JSONDecodeError, TypeError) as json_error:
                        logger.warning(
                            f"JSON 解析失败 ({json_error})，尝试直接使用字符串"
                        )
                        # 如果不是 JSON，尝试直接使用字符串
                        # 但对于有 args_schema 的工具，这通常会失败
                        try:
                            result = await original_tool.ainvoke(tool_input)
                        except Exception as str_error:
                            logger.error(f"字符串输入也失败: {str_error}")
                            # 尝试将字符串包装为常见的参数格式
                            fallback_formats = [
                                {"query": tool_input},
                                {"city": tool_input},
                                {"input": tool_input},
                                {"text": tool_input},
                            ]

                            for fallback_format in fallback_formats:
                                try:
                                    logger.info(f"尝试备用格式: {fallback_format}")
                                    result = await original_tool.ainvoke(
                                        fallback_format
                                    )
                                    logger.info(f"备用格式成功: {fallback_format}")
                                    break
                                except Exception:
                                    continue
                            else:
                                # 所有格式都失败
                                raise str_error
                else:
                    # 没有 schema 的工具直接使用字符串输入
                    logger.info(
                        f"工具 {original_tool.name} 无 args_schema，直接使用字符串"
                    )
                    result = await original_tool.ainvoke(tool_input)

                # 确保返回字符串格式
                result_str = (
                    str(result) if result is not None else "工具执行完成，但无返回结果"
                )
                logger.info(
                    f"工具 {original_tool.name} 执行成功，返回结果长度: {len(result_str)}"
                )
                return result_str

            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                logger.error(f"工具 {original_tool.name} 执行失败: {e}")
                return error_msg

        # 同步版本的包装函数
        def sync_wrapped_tool_func(tool_input: str) -> str:
            """同步版本的工具包装器"""
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(wrapped_tool_func(tool_input))
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                return asyncio.run(wrapped_tool_func(tool_input))

        # 创建新的工具实例，兼容 ReAct Agent
        return Tool(
            name=original_tool.name,
            description=original_tool.description,
            func=sync_wrapped_tool_func,  # 同步版本
            coroutine=wrapped_tool_func,  # 异步版本
        )

    async def _get_all_tools(self) -> List[Any]:
        """获取所有可用工具（基础工具 + MCP 工具），并为 ReAct Agent 创建兼容包装器"""
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

        # 获取原始工具列表
        original_tools = self.base_tools + mcp_tools

        # 为 ReAct Agent 创建兼容的工具包装器
        compatible_tools = []
        for tool in original_tools:
            try:
                # 为所有工具创建包装器，以确保一致的行为和错误处理
                logger.info(f"为工具 {tool.name} 创建 ReAct 兼容包装器")
                logger.info(f"  工具类型: {type(tool)}")
                logger.info(
                    f"  有 args_schema: {hasattr(tool, 'args_schema') and tool.args_schema is not None}"
                )

                wrapped_tool = self._create_react_compatible_tool_wrapper(tool)
                compatible_tools.append(wrapped_tool)

            except Exception as e:
                logger.error(f"为工具 {tool.name} 创建包装器失败: {e}")
                logger.error("错误详情:", exc_info=True)
                # 如果包装失败，仍尝试使用原工具
                logger.warning(f"使用原始工具 {tool.name}")
                compatible_tools.append(tool)

        logger.info("--- All Tools for LangChain ReAct Agent (兼容处理后) ---")
        for tool in compatible_tools:
            logger.info(f"  Name: {tool.name}")
        logger.info("-" * 40)

        return compatible_tools

    async def stream_chat(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户输入，使用 ReAct Agent 实现推理-行动模式

        Args:
            user_input: 用户输入的问题
            session_id: 会话ID，用于历史记录管理

        Yields:
            Dict[str, Any]: 包含 'type' 和 'data' 的事件字典
            - {"type": "chunk", "data": "token"}  # 流式输出
            - {"type": "thought", "data": "思考过程"}  # ReAct 特有：思考过程
            - {"type": "action", "data": {...}}  # ReAct 特有：行动决策
            - {"type": "tool_call", "data": {...}}  # 工具调用事件
            - {"type": "tool_result", "data": {...}}  # 工具结果事件
            - {"type": "observation", "data": "观察结果"}  # ReAct 特有：观察结果
            - {"type": "stream_end"}  # 流结束标识
        """
        try:
            logger.info(f"开始处理用户输入: '{user_input}' (session_id: {session_id})")

            # 获取 LLM - 与原版本相同的配置
            llm = get_llms(supplier="volces", model="deepseek")

            # 验证 LLM 是否支持流式输出
            logger.info(f"LLM 流式输出配置: {getattr(llm, 'streaming', 'unknown')}")

            # 获取所有工具
            all_tools = await self._get_all_tools()

            # 创建 ReAct Agent - 核心差异点
            agent = create_react_agent(llm, all_tools, self.react_prompt)

            # 创建 AgentExecutor - 配置与原版本类似
            agent_executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=True,  # ReAct Agent 建议开启详细输出以显示思考过程
                handle_parsing_errors=True,
                max_iterations=10,
                return_intermediate_steps=True,
                streaming=True,
            )
            logger.info("ReAct AgentExecutor 创建成功")

            # 包装历史记录管理 - 针对 ReAct Agent 进行调整
            # 注意：ReAct Agent 使用字符串格式的历史记录，但 RunnableWithMessageHistory 会自动处理转换
            agent_with_history = RunnableWithMessageHistory(
                agent_executor,
                self.get_session_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",  # ReAct 模板不使用这个，但保持一致性
                output_messages_key="output",
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

            logger.info(f"开始流式执行 ReAct agent (session_id: {session_id})")

            # 用于收集完整输出
            collected_output = []

            # 使用 astream_events 进行流式处理 - 事件处理需要适配 ReAct 模式
            async for event in agent_with_history.astream_events(
                agent_input, config=config
            ):
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                # ReAct Agent 特有的事件处理

                # 处理 LLM 流式输出 - 包含思考过程
                if event_type == "on_llm_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content
                        if token:
                            collected_output.append(token)

                            # ReAct 特有：识别思考过程
                            if "Thought:" in token or "思考:" in token:
                                yield {"type": "thought", "data": token}
                            elif "Action:" in token or "行动:" in token:
                                yield {"type": "action", "data": token}
                            # elif "Observation:" in token or "观察:" in token:
                            # yield {"type": "observation", "data": token}
                            else:
                                yield {"type": "chunk", "data": token}

                # 处理 Chat Model 流式输出
                elif event_type == "on_chat_model_stream":
                    chunk = event_data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content
                        if token:
                            collected_output.append(token)
                            yield {"type": "chunk", "data": token}

                # 处理工具调用 - 与原版本相同的逻辑
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event_data.get("input", {})
                    run_id = event.get("run_id", "")
                    call_id = (
                        f"call_{run_id}_{tool_name}"
                        if run_id
                        else f"call_{hash(str(tool_name) + str(tool_input))}"
                    )

                    if not hasattr(self, "_tool_call_mapping"):
                        self._tool_call_mapping = {}
                    mapping_key = f"{run_id}_{tool_name}" if run_id else tool_name
                    self._tool_call_mapping[mapping_key] = call_id

                    logger.info(
                        f"ReAct Agent 工具调用: {tool_name}, call_id: {call_id}"
                    )
                    yield {
                        "type": "tool_call",
                        "data": {"name": tool_name, "args": tool_input, "id": call_id},
                    }

                # 处理工具调用结束 - 与原版本相同的逻辑
                elif event_type == "on_tool_end":
                    tool_name = event_name
                    tool_output = event_data.get("output")
                    run_id = event.get("run_id", "")

                    mapping_key = f"{run_id}_{tool_name}" if run_id else tool_name
                    call_id = getattr(self, "_tool_call_mapping", {}).get(
                        mapping_key, f"call_{hash(str(tool_name))}"
                    )

                    logger.info(
                        f"ReAct Agent 工具结果: {tool_name}, tool_call_id: {call_id}"
                    )
                    yield {
                        "type": "tool_result",
                        "data": {
                            "name": tool_name,
                            "content": str(tool_output),
                            "tool_call_id": call_id,
                        },
                    }

                # 处理其他流式事件 - 与原版本相同的逻辑
                elif "stream" in event_type and event_data:
                    chunk = event_data.get("chunk")
                    if chunk:
                        content = None
                        if hasattr(chunk, "content"):
                            content = chunk.content
                        elif isinstance(chunk, str):
                            content = chunk
                        elif isinstance(chunk, dict) and "content" in chunk:
                            content = chunk["content"]

                        if content and content.strip():
                            collected_output.append(content)
                            yield {"type": "chunk", "data": content}

            # 发送流结束信号
            yield {"type": "stream_end"}
            logger.info(f"ReAct Agent 流式处理完成 (session_id: {session_id})")

        except Exception as e:
            logger.error(
                f"ReAct Agent 流式处理时发生错误 (session_id: {session_id}): {e}",
                exc_info=True,
            )
            yield {"type": "error", "data": f"处理请求时发生错误: {e}"}


# 全局实例变量（懒加载）
_langchain_react_agent_instance = None


def get_langchain_react_agent():
    """获取 LangChain ReAct Agent 实例（懒加载模式）"""
    global _langchain_react_agent_instance
    if _langchain_react_agent_instance is None:
        _langchain_react_agent_instance = LangChainReActAgent()
    return _langchain_react_agent_instance


async def main_graph_execution(
    user_input: str,
    session_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    主要执行函数，保持与原版本相同的接口

    Args:
        user_input: 用户输入
        session_id: 会话ID

    Yields:
        Dict[str, Any]: 事件流字典，包含以下类型：
        - {"type": "thought", "data": "..."}      # ReAct 特有：思考过程
        - {"type": "action", "data": "..."}       # ReAct 特有：行动决策
        - {"type": "observation", "data": "..."}  # ReAct 特有：观察结果
        - {"type": "tool_call", "data": {...}}    # 工具调用开始
        - {"type": "tool_result", "data": {...}}  # 工具执行结果
        - {"type": "chunk", "data": "..."}        # 流式文本输出
        - {"type": "stream_end"}                  # 流结束标识
        - {"type": "error", "data": "..."}        # 错误信息
    """
    react_agent = get_langchain_react_agent()
    async for event in react_agent.stream_chat(user_input, session_id):
        yield event


# Integration tests have been migrated to test/integration/service/test_langchain_react_agent.py
# Run with: uv run pytest test/integration/service/test_langchain_react_agent.py -v
