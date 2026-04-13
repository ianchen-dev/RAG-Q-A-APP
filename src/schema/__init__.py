"""Schema package for request/response models."""

from src.schema.agent import QueryRequest
from src.schema.chat import (
    ChatConfig,
    ChatRequest,
    KnowledgeConfig,
    LLMConfig,
    RerankerConfig,
)
from src.schema.health import (
    ConnectionStatsResponse,
    HealthCheckResponse,
    OneAPIHealthResponse,
)
from src.schema.knowledge import KnowledgeBaseCreate

__all__ = [
    # Chat schemas
    "LLMConfig",
    "RerankerConfig",
    "KnowledgeConfig",
    "ChatConfig",
    "ChatRequest",
    # Health schemas
    "HealthCheckResponse",
    "ConnectionStatsResponse",
    "OneAPIHealthResponse",
    # Knowledge schemas
    "KnowledgeBaseCreate",
    # Agent schemas
    "QueryRequest",
]
