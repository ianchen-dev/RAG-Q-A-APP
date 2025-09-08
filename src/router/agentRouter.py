import json  # 添加 json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 并行部署：同时支持两种 Agent 类型
# from src.service.langgraph_agent import main_graph_execution  # 原 LangGraph 版本
from src.service.langchain_agent import main_graph_execution as tool_calling_execution  # Tool Calling Agent
from src.service.langchain_react_agent import main_graph_execution as react_execution  # ReAct Agent

AgentRouter = APIRouter()


# 2. 定义一个请求体模型
class QueryRequest(BaseModel):
    question: str  # 用户输入的消息
    session_id: str = "67fa7b1acaaf230867eefce1"


async def sse_event_formatter_agent(
    query_request: QueryRequest,
    execution_func,
) -> AsyncGenerator[str, None]:
    """通用的 SSE 事件格式化器，支持不同类型的 Agent"""
    try:
        async for event_dict in execution_func(
            query_request.question, query_request.session_id
        ):
            yield f"data: {json.dumps(event_dict)}\n\n"
    except Exception as e:
        # Log the exception here if you have a logger configured
        print(f"Error during SSE formatting for agent: {e}")  # Basic error logging
        error_payload = json.dumps(
            {"type": "error", "data": f"Agent stream processing error: {e}"}
        )
        yield f"data: {error_payload}\n\n"
    finally:
        print("SSE event formatter for agent finished.")


@AgentRouter.post("/tool-calling")
async def query_tool_calling_agent(query_request: QueryRequest) -> StreamingResponse:
    """
    使用 Tool Calling Agent 处理查询请求并以流式方式返回响应。
    
    Tool Calling Agent 特点：
    - 支持原生工具调用
    - 更简洁的对话模式
    - 适合快速响应场景

    Args:
        query_request (QueryRequest): 包含用户查询和会话ID的请求体

    Returns:
        StreamingResponse: 包含 SSE 事件的流式响应
        事件类型包括：chunk, tool_call, tool_result, stream_end, error
    """
    return StreamingResponse(
        sse_event_formatter_agent(query_request, tool_calling_execution), 
        media_type="text/event-stream"
    )


@AgentRouter.post("/mcp")
async def query_react_agent(query_request: QueryRequest) -> StreamingResponse:
    """
    使用 ReAct Agent 处理查询请求并以流式方式返回响应。
    
    ReAct Agent 特点：
    - 支持推理-行动模式（Reasoning and Acting）
    - 显示详细的思考过程
    - 适合需要复杂推理的场景

    Args:
        query_request (QueryRequest): 包含用户查询和会话ID的请求体

    Returns:
        StreamingResponse: 包含 SSE 事件的流式响应
        事件类型包括：thought, action, observation, tool_call, tool_result, chunk, stream_end, error
    """
    return StreamingResponse(
        sse_event_formatter_agent(query_request, react_execution), 
        media_type="text/event-stream"
    )


# @AgentRouter.post("/mcp")  
# async def query_mcp_stream(query_request: QueryRequest) -> StreamingResponse:
#     """
#     兼容性接口：默认使用 Tool Calling Agent 处理 MCP 查询请求。
    
#     注意：这是为了保持向后兼容性而保留的接口。
#     建议使用具体的 /tool-calling 或 /react 端点。

#     Args:
#         query_request (QueryRequest): 包含用户查询和会话ID的请求体

#     Returns:
#         StreamingResponse: 包含 SSE 事件的流式响应
#     """
#     return StreamingResponse(
#         sse_event_formatter_agent(query_request, tool_calling_execution), 
#         media_type="text/event-stream"
#     )


# 保留旧的非流式接口作为参考或备用，如果需要的话
# @AgentRouter.post("/mcp_non_stream")
# async def query_mcp_non_stream(user_input: str) -> Dict[str, Any]:
#     """
#     处理 MCP 查询请求 (非流式)
#     """
#     # 假设有一个非流式的 main_graph_execution_non_stream 版本
#     # response = await main_graph_execution_non_stream(user_input)
#     # return {"response": response, "status": "success"}
#     pass # 暂时留空
