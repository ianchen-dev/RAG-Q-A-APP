"""Health check schemas for request/response models."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="总体状态")
    timestamp: str = Field(..., description="检查时间")
    services: Dict[str, Any] = Field(..., description="各服务状态")


class ConnectionStatsResponse(BaseModel):
    """Connection statistics response model."""

    mongodb: Dict[str, Any] = Field(..., description="MongoDB 连接信息")


class OneAPIHealthResponse(BaseModel):
    """OneAPI health check response model."""

    overall_status: str = Field(..., description="总体状态")
    timestamp: str = Field(..., description="检查时间")
    connection: Dict[str, Any] = Field(..., description="连接检查结果")
    embeddings: Optional[Dict[str, Any]] = Field(None, description="嵌入模型检查结果")
