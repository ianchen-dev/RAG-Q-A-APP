import logging
import os
from typing import Any, AsyncGenerator, Dict  # 修改类型提示

from langchain_core.messages import (  # 添加 AIMessage 和 ToolMessage 导入
    AIMessage,
    HumanMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# Import tools from the new src.tools package
from langgraph.prebuilt import create_react_agent

from src.components.llm_provider import get_llms

# --- 从新的管理器导入MCP工具获取函数 ---
from src.config.mcp_client_manager import get_cached_mcp_tools
from src.tools import (
    create_tavily_tool,
    get_knowledge_list_tool,
    retriever_document_tool,
)

# 基础工具配置 - 条件性添加 Tavily 搜索工具
base_tools = [get_knowledge_list_tool, retriever_document_tool]

# 如果 TavilySearch 可用，则添加到工具列表
tavily_tool = create_tavily_tool(max_results=2)
if tavily_tool is not None:
    base_tools.append(tavily_tool)
    logger.info("成功添加 TavilySearch 工具")
else:
    logger.info("TavilySearch 不可用，仅使用知识库工具")

# --- checkpoint 持久化 ---
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # MongoDB 检查点器

from src.config.database_manager import (  # 从数据库管理器获取数据库实例
    get_database_manager,
)


async def main_graph_execution(
    user_input: str,
    session_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:  # 修改返回类型为字典的异步生成器
    """
    Processes events from the react agent graph and yields them as dictionaries.
    """
    # MCP客户端配置 (已移至 mcp_client_manager.py, 此处保留注释备查)
    # mcp_config = {
    #     "howtocook-mcp": {"command": "npx", "args": ["-y", "howtocook-mcp"]},
    #     "amap-maps": {
    #         "command": "npx",
    #         "args": ["-y", "@amap/amap-maps-mcp-server"],
    #         "env": {"AMAP_MAPS_API_KEY": "d0ba669a99119b0245cdc0f26cc45607"},
    #     },
    #     # "current_datetime": {
    #     #     "command": "mcp-proxy",
    #     #     "args": ["http://127.0.0.1:8001/current_datetime"],
    #     # },
    #     # 可以添加更多MCP服务配置
    # }

    # --- 获取 MongoDB 连接信息并准备 MongoDBSaver ---
    try:
        manager = await get_database_manager()
        motor_database_instance = await manager.get_mongodb_database()
        db_name = motor_database_instance.name
        logger.info(f"成功获取 MongoDB 数据库实例: {db_name}")
    except RuntimeError as e:
        logger.error(f"获取 MongoDB 实例失败: {e}")
        yield {"type": "error", "data": f"MongoDB not properly initialized: {e}"}
        return

    # 获取 MongoDB 连接字符串
    mongodb_url = os.getenv("MONGODB_URL")
    if not mongodb_url:
        logger.error("MONGODB_URL 环境变量未设置")
        yield {"type": "error", "data": "MONGODB_URL environment variable not set"}
        return

    checkpoint_collection_name = (
        "checkpointsHistory"  # 集合名称（注意：新版本参数名去掉了 s）
    )
    logger.info(
        f"准备初始化 MongoDBSaver，数据库: {db_name}, 集合: {checkpoint_collection_name}"
    )
    # --- MongoDB 配置结束 ---

    # async with MultiServerMCPClient(mcp_config) as client: # 注释掉，改用管理器获取工具
    #     logger.info("MCP Client started.")
    #     llm = get_llms(
    #         supplier="oneapi",
    #         model="deepseek-ai/DeepSeek-V3",  # 确保模型有效支持工具调用
    #         api_key=os.getenv("ONEAPI_API_KEY"),
    #         streaming=True,
    #     )
    #     mcp_tools = client.get_tools()
    #     logger.info("--- Acquired MCP Tools ---")
    #     for m_tool in mcp_tools:
    #         logger.info(f"  Name: {m_tool.name}, Description: {m_tool.description}")
    #     logger.info("-" * 26)

    #     all_tools_for_this_run = base_tools + mcp_tools
    #     logger.info("--- All Tools for this Run ---")
    #     for t in all_tools_for_this_run:
    #         logger.info(f"  Name: {t.name}")
    #     logger.info("-" * 30)

    #     # 使用 MongoDBSaver 替换 MemorySaver
    #     agent_executor = create_react_agent(
    #         model=llm,
    #         tools=all_tools_for_this_run,
    #         checkpointer=checkpointer,  # <--- 使用配置好的 MongoDB checkpointer
    #         # debug=True # 可选
    #     )
    #     logger.info(
    #         "Graph compiled using create_react_agent with MongoDB checkpointer."
    #     )

    #     # 使用传入的 session_id 作为 thread_id
    #     config = {"configurable": {"thread_id": session_id}}
    #     logger.info(
    #         f"Streaming graph with input: '{user_input}' for thread_id: '{session_id}'"
    #     )

    #     # 使用 astream 并指定 stream_mode=["updates", "messages"]
    #     # LangGraph 的 astream 在组合模式下通常 yield (stream_mode, event_data)
    #     async for stream_mode, event_data in agent_executor.astream(
    #         {"messages": [{"role": "user", "content": user_input}]},
    #         config,
    #         stream_mode=["updates", "messages"],
    #     ):

    # 使用新版本的 AsyncMongoDBSaver.from_conn_string 异步上下文管理器
    async with AsyncMongoDBSaver.from_conn_string(
        conn_string=mongodb_url,
        db_name=db_name,
        checkpoint_collection_name=checkpoint_collection_name,
    ) as checkpointer:
        logger.info(f"MongoDBSaver 初始化成功，集合: {checkpoint_collection_name}")

        # 从管理器获取预加载的 MCP 工具
        mcp_tools = await get_cached_mcp_tools()
        if not mcp_tools:
            # 这是一个关键的检查点：如果应用启动时MCP客户端未能成功加载工具，
            # 那么在这里它们将为空。需要决定此时的行为：
            # 1. 记录警告/错误并继续，只使用 base_tools (Agent能力受限)
            # 2. 抛出异常或返回错误信息，指示MCP工具不可用
            # 3. 尝试在此处动态启动一个临时的MCPClient (违背初衷，但作为后备)
            logger.warning(
                "MCP tools not available from manager. Proceeding without them or with an error."
            )
            # 选项1: 继续，可能只用base_tools，或者让 all_tools_for_this_run 为空导致后续问题
            # 或者，如果MCP工具是必须的，应该在这里处理错误并提前返回
            # yield {"type": "error", "data": "Critical MCP tools are not available."}
            # return

        logger.info(f"Retrieved {len(mcp_tools)} MCP tools from manager.")
        for m_tool in mcp_tools:
            logger.info(f"  MCP Tool (from manager): {m_tool.name}")

        all_tools_for_this_run = base_tools + mcp_tools
        logger.info("--- All Tools for this Run (using managed MCP tools) ---")
        for t in all_tools_for_this_run:
            logger.info(f"  Name: {t.name}")
        logger.info("-" * 30)
        # 获取 LLM

        llm = get_llms(
            supplier="oneapi",
            model="deepseek-ai/DeepSeek-V3",  # 确保模型有效支持工具调用
            api_key=os.getenv("ONEAPI_API_KEY"),
            streaming=True,
        )
        # llm = get_llms(
        #     supplier="siliconflow",
        #     model="deepseek-ai/DeepSeek-V3",
        #     api_key=os.getenv("SILICONFLOW_API_KEY"),
        #     streaming=True,
        # )

        # 使用 MongoDBSaver 替换 MemorySaver
        agent_executor = create_react_agent(
            model=llm,
            tools=all_tools_for_this_run,
            checkpointer=checkpointer,  # <--- 使用配置好的 MongoDB checkpointer
            # debug=True # 可选
        )
        logger.info(
            "Graph compiled using create_react_agent with MongoDB checkpointer (and managed MCP tools)."
        )

        # 使用传入的 session_id 作为 thread_id
        config = {"configurable": {"thread_id": session_id}}
        logger.info(
            f"Streaming graph with input: '{user_input}' for thread_id: '{session_id}' (managed MCP)"
        )

        async for stream_mode, event_data in agent_executor.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode=["updates", "messages"],
        ):
            # print(f"Stream mode: {stream_mode}, Event data: {event_data}")
            if stream_mode == "messages":
                token = ""
                if isinstance(event_data, list) and event_data:
                    message_chunk = event_data[0]
                    if hasattr(message_chunk, "content"):
                        token = message_chunk.content
                elif isinstance(event_data, tuple) and len(event_data) > 0:
                    message_chunk = event_data[0]
                    if hasattr(message_chunk, "content"):
                        token = message_chunk.content
                elif isinstance(event_data, str):
                    token = event_data
                # logger.info(
                #     f"Stream mode: {stream_mode},token.type: {type(token)}, Event data: {token}"
                # )
                if token.startswith(("{", "[")):  # 过滤掉其中的 ToolMessage
                    token = ""
                if token:
                    yield {"type": "chunk", "data": token}  # Yield 字典

            elif stream_mode == "updates":
                logger.info(
                    f"___updates______Stream mode: {stream_mode},type: {type(event_data)}, Event data: {event_data}"
                )
                if (
                    isinstance(event_data, dict)
                    and "agent" in event_data
                    and isinstance(event_data["agent"], dict)
                    and "messages" in event_data["agent"]
                    and event_data["agent"]["messages"]
                ):
                    last_message = event_data["agent"]["messages"][-1]
                    logger.info(
                        f"Stream mode: {stream_mode},last_message.type: {type(last_message)}, Event data: {last_message}"
                    )
                    if isinstance(last_message, AIMessage) and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_call_event_data = {
                                "name": tool_call["name"],
                                "args": tool_call["args"],
                                "id": tool_call["id"],
                            }
                            yield {
                                "type": "tool_call",
                                "data": tool_call_event_data,
                            }  # Yield 字典
                    elif isinstance(last_message, ToolMessage):
                        tool_result_event_data = {
                            "name": last_message.name,
                            "content": str(last_message.content),
                            "tool_call_id": last_message.tool_call_id,
                        }
                        yield {
                            "type": "tool_result",
                            "data": tool_result_event_data,
                        }  # Yield 字典

                # Check for messages under 'tools' key, as observed in logs for direct tool outputs
                if (
                    isinstance(event_data, dict)
                    and "tools" in event_data
                    and isinstance(event_data["tools"], dict)
                    and "messages" in event_data["tools"]
                    and event_data["tools"]["messages"]
                ):
                    for message_item in event_data["tools"]["messages"]:
                        if isinstance(message_item, ToolMessage):
                            logger.info(
                                f"Stream mode: {stream_mode} (tools path), tool_message.type: {type(message_item)}, Event data: {message_item}"
                            )
                            tool_result_event_data = {
                                "name": message_item.name,
                                "content": str(
                                    message_item.content
                                ),  # Ensure content is string
                                "tool_call_id": message_item.tool_call_id,
                            }
                            yield {
                                "type": "tool_result",
                                "data": tool_result_event_data,
                            }  # Yield 字典

        yield {"type": "stream_end"}  # Yield 字典
        logger.info(f"Graph streaming finished for thread_id: '{session_id}'.")
