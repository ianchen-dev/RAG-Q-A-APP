from typing import List

from beanie import PydanticObjectId
from fastapi import HTTPException
from pydantic import BaseModel

from src.models.assistant import Assistant
from src.models.session import Session
from src.service.ChatSev import ChatSev


class AssistantRequest(BaseModel):
    title: str = "新助手"
    username: str
    prompt: str = "你是一个AI助手，请根据用户的问题给出回答。"
    knowledge_Id: str | None = None


async def create_assistant(assistant_data: AssistantRequest) -> Assistant:
    """
    创建新的助手。

    Args:
        assistant_data: 包含助手信息的请求体。
            - title: 助手标题
            - username: 用户名
            - prompt: 助手提示词 (可选)
            - knowledge_Id: 知识库ID (可选)

    Returns:
        返回创建的 Assistant 文档对象。

    Raises:
        HTTPException: 如果已存在同名用户名的助手 (虽然模型是 unique，插入时会报错，这里可以提前检查或依赖数据库报错).
                       在实际应用中，可能需要更复杂的错误处理。
    """
    assistant_doc = Assistant(
        title=assistant_data.title,
        username=assistant_data.username,
        prompt=assistant_data.prompt,
        knowledge_Id=assistant_data.knowledge_Id,
    )
    await assistant_doc.insert()
    return assistant_doc


async def get_assistant_list(username: str) -> List[Assistant]:
    """
    根据用户名获取该用户的所有助手列表。

    Args:
        username: 用户名。

    Returns:
        该用户的 Assistant 文档对象列表，按创建时间降序排序。
    """
    assistants = (
        await Assistant.find(Assistant.username == username)
        .sort(Assistant.created_at)
        .to_list()
    )
    return assistants


async def update_assistant(
    assistant_id: str, assistant_data: AssistantRequest
) -> Assistant:
    """
    根据助手 ID 更新指定助手的信息。
    只更新 title, prompt, knowledge_Id。

    Args:
        assistant_id: 要更新的助手的 ID (来自路径参数)。
        assistant_data: 包含要更新的助手信息的请求体。
            - title: 新的助手标题
            - username: (此字段在更新时被忽略)
            - prompt: 新的助手提示词 (可选)
            - knowledge_Id: 新的知识库ID (可选)

    Returns:
        更新后的 Assistant 文档对象。

    Raises:
        HTTPException: 如果找不到指定 ID 的助手 (404) 或 ID 格式无效 (400)。
    """
    try:
        assistant_object_id = PydanticObjectId(assistant_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的助手 ID 格式")

    assistant = await Assistant.get(assistant_object_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="找不到指定的助手")

    assistant.title = assistant_data.title
    assistant.prompt = assistant_data.prompt
    assistant.knowledge_Id = assistant_data.knowledge_Id

    await assistant.save()
    return assistant


async def delete_assistant(assistant_id: str) -> dict:
    """
    根据助手 ID 删除指定的助手及其关联的所有会话和聊天记录。

    Args:
        assistant_id: 要删除的助手的 ID (来自查询参数)。

    Returns:
        一个包含成功消息的字典。

    Raises:
        HTTPException: 如果找不到指定 ID 的助手 (404) 或 ID 格式无效 (400)。
    """
    try:
        assistant_object_id = PydanticObjectId(assistant_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的助手 ID 格式")

    assistant = await Assistant.get(assistant_object_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="找不到指定的助手")

    associated_sessions = await Session.find(
        Session.assistant_id == assistant_id
    ).to_list()

    chat_service = ChatSev(knowledge=None)
    for session in associated_sessions:
        try:
            await chat_service.clear_history(str(session.id))
            await session.delete()
        except Exception as e:
            print(f"删除会话 {session.id} 或其历史记录时出错: {e}")

    await assistant.delete()

    return {"message": "助手及关联会话已成功删除"}


class SessionCreate(BaseModel):
    username: str = "root"
    assistant_id: str
    assistant_name: str = "Default Assistant"


async def create_session(
    session_data: SessionCreate,
):
    """
    用户创建新的会话

    Args:
        title: 会话标题。
        username: 用户名。
        assistant_id: 助手 ID。
        assistant_name: 助手名称。

    Returns:
        返回创建的文档对象。
    """
    session_doc: Session = Session(
        title=session_data.title,
        username=session_data.username,
        assistant_id=session_data.assistant_id,
        assistant_name=session_data.assistant_name,
    )
    await session_doc.insert()
    return session_doc


async def get_session_list(username: str) -> List[Session]:
    """
    根据用户名获取该用户的所有会话列表。

    Args:
        username: 用户名。

    Returns:
        该用户的 Session 文档对象列表。
    """
    sessions = (
        await Session.find(Session.username == username).sort(-Session.date).to_list()
    )
    return sessions


async def update_session_title(session_id: str, title: str) -> Session:
    """
    根据会话 ID 修改指定会话的标题。

    Args:
        session_id: 要修改的会话的 ID。
        title: 新的会话标题。

    Returns:
        更新后的 Session 文档对象。

    Raises:
        HTTPException: 如果找不到指定 ID 的会话或 ID 格式无效。
    """
    try:
        session_object_id = PydanticObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的会话 ID 格式")

    session = await Session.get(session_object_id)
    if not session:
        raise HTTPException(status_code=404, detail="找不到指定的会话")

    session.title = title
    await session.save()
    return session


async def delete_session(session_id: str) -> dict:
    """
    根据会话 ID 删除指定的会话及其聊天记录。

    Args:
        session_id: 要删除的会话的 ID。

    Returns:
        一个包含成功消息的字典。

    Raises:
        HTTPException: 如果找不到指定 ID 的会话或 ID 格式无效。
    """
    try:
        session_object_id = PydanticObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的会话 ID 格式")

    session = await Session.get(session_object_id)
    if not session:
        raise HTTPException(status_code=404, detail="找不到指定的会话")
    await session.delete()

    try:
        ChatSev(knowledge=None).clear_history(session_id)
    except Exception as e:
        print(f"警告：清除会话 {session_id} 的历史记录时出错: {e}")

    return {"message": "会话及其历史记录已成功删除"}
