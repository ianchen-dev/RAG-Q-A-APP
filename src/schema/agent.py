"""Agent schemas for request/response models."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Agent query request model."""

    question: str  # 用户输入的消息
    session_id: str = "67fa7b1acaaf230867eefce1"
