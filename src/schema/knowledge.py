"""Knowledge base schemas for request/response models."""

from typing import Optional

from pydantic import BaseModel

from src.models.knowledgeBase import EmbeddingConfig


class KnowledgeBaseCreate(BaseModel):
    """Knowledge base creation request model."""

    title: str
    tag: Optional[list[str]] = None
    description: Optional[str] = None
    embedding_config: EmbeddingConfig  # 直接使用 EmbeddingConfig 模型
