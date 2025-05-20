import json  # 添加 json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# from src.service.mcp_graph import main_graph_execution
from src.service.react_agent import main_graph_execution

AgentRouter = APIRouter()


# 2. 定义一个请求体模型
class QueryRequest(BaseModel):
    question: str
    session_id: str
    # 如果将来需要更多参数，可以在这里添加
    # user_id: str | None = None


async def sse_event_formatter_agent(
    query_request: QueryRequest,
) -> AsyncGenerator[str, None]:
    """Wraps main_graph_execution to format its dict output into SSE event strings."""
    try:
        async for event_dict in main_graph_execution(
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


@AgentRouter.post("/mcp")
async def query_mcp_stream(query_request: QueryRequest) -> StreamingResponse:
    """
    处理 MCP 查询请求并以流式方式返回响应。

    Args:
        user_input (str): 用户的查询文本

    Returns:
        StreamingResponse: 包含 SSE 事件的流式响应

    Raises:
        HTTPException: 当查询处理失败时抛出 (此处的错误处理需要调整以适应流式响应)
    """
    return StreamingResponse(
        sse_event_formatter_agent(query_request), media_type="text/event-stream"
    )


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
