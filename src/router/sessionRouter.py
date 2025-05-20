from fastapi import APIRouter

import src.service.sessionSev as sessionSev

SessionRouter = APIRouter()


# 新建会话
@SessionRouter.post("/create", summary="新建会话")
async def create_session(session: sessionSev.SessionCreate):
    return await sessionSev.create_session(session)


# 获取用户会话列表
@SessionRouter.get("/list", summary="获取用户会话列表")
async def get_session_list(username: str, assistant_id: str):
    return await sessionSev.get_session_list(username, assistant_id)


# 修改会话标题
@SessionRouter.put("/{session_id}/title", summary="修改会话标题")
async def update_session_title(session_id: str, title: str):
    return await sessionSev.update_session_title(session_id, title)


# 删除会话-并且一并删除会话中的历史消息
@SessionRouter.delete("/{session_id}/delete", summary="删除会话")
async def delete_session(session_id: str):
    return await sessionSev.delete_session(session_id)


# --- 分页获取历史记录接口 ---
@SessionRouter.get("/{session_id}/history", summary="分页获取会话历史记录")
async def get_session_history_endpoint(
    session_id: str, page: int = 1, page_size: int = 10
):
    """
    成功响应 (200 OK):
        {
    "page": 1,
    "size": 5, // 当前页实际返回的消息数量
    "total_pages": 3, // 总页数
    "total_items": 12, // 该会话总消息数
    "items": [
        {
        "type": "ai",
        "content": "这是最新的 AI 回复。"
        // "timestamp": "2023-10-27T10:30:00Z" // 如果添加了时间戳字段
        },
        {
        "type": "human",
        "content": "这是用户最新的问题。"
        },
        // ... more messages for the current page ...
    ]
    }
    错误响应:
    400 Bad Request: 如果 session_id 格式无效。
        {
      "detail": "无效的会话 ID 格式"
        }
    500 Internal Server Error: 如果服务器在处理请求时遇到内部错误（例如数据库连接问题）。
        {
      "detail": "获取历史记录时发生内部错误"
        }
    """

    # 注意：FastAPI 会自动从 Query 参数获取 page 和 page_size
    # 确保 FastAPI 版本支持 Query 参数的默认值或显式使用 Query
    # from fastapi import Query # 如果需要显式导入
    # async def get_session_history_endpoint(
    #     session_id: str,
    #     page: int = Query(1, ge=1, description="页码，从1开始"),
    #     page_size: int = Query(10, ge=1, le=100, description="每页数量")
    # ):
    return await sessionSev.get_session_history(session_id, page, page_size)
