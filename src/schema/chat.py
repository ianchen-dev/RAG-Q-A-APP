"""Chat schemas for request/response models."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration model."""

    supplier: Literal["ollma", "openai", "siliconflow", "oneapi"] = "oneapi"
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    api_key: str
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)


class RerankerConfig(BaseModel):
    """Reranker configuration model."""

    use_reranker: bool = Field(default=False, description="是否启用重排序")
    reranker_type: Literal["local", "remote"] = Field(
        default="remote",
        description="重排序器类型: 'local' (本地CrossEncoder) 或 'remote' (SiliconFlow API)",
    )
    remote_rerank_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="远程 Reranker (SiliconFlow) 配置，当 reranker_type='remote' 时需要。至少包含 'api_key' 和可选的 'model'。 例如: {'api_key': 'your_sf_key', 'model': 'BAAI/bge-reranker-v2-m3'}",
    )
    rerank_top_n: int = Field(default=3, ge=1, description="重排序后返回的文档数量")


class KnowledgeConfig(BaseModel):
    """Knowledge base configuration for retrieval."""

    knowledge_base_id: str
    filter_by_file_md5: Optional[list[str]] = Field(
        default=None,
        description="文件MD5，用于指定检索的若干个文档，如果为None，则检索所选知识库中的全部文件",
    )
    search_k: Optional[int] = Field(
        default=10, ge=1, description="基础检索器返回的文档数量 (应 >= rerank_top_n)"
    )
    # --- BM25 相关配置 ---
    use_bm25: bool = Field(default=False, description="是否启用 BM25 混合检索")
    bm25_k: int = Field(default=3, ge=1, description="BM25 检索器返回的文档数量")
    # --- 重排序器配置 ---
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig, description="重排序器配置"
    )


class ChatConfig(BaseModel):
    """Chat configuration model."""

    chat_history_max_length: Optional[int] = Field(default=8, ge=0)
    prompt_override: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""

    question: str = "你好"
    session_id: str
    llm_config: LLMConfig
    chat_config: Optional[ChatConfig] = Field(default_factory=ChatConfig)
    knowledge_config: Optional[KnowledgeConfig] = None
