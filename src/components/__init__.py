"""Components package for RAG ChatBot.

This package contains reusable components for the application,
such as prompt builders, chain builders, chat history managers,
stream handlers, and other utilities.
"""

from src.components.chain_builder import ChainBuilder
from src.components.chat_history import ChatHistoryManager
from src.components.embedding_provider import EmbeddingSupplier, create_embedding, get_embedding
from src.components.kb import (
    DocumentProcessor,
    FileProcessor,
    KnowledgeBaseFactory,
    KnowledgeBaseRepository,
    KnowledgeBaseValidator,
    KnowledgeManager,
    RetrieverBuilder,
    VectorDBManager,
)
from src.components.llm_provider import LLMSupplier, create_llm, get_llms
from src.components.prompt import create_chat_prompts
from src.components.reranker_compressor import RemoteRerankerCompressor
from src.components.stream_handler import StreamHandler
from src.components.vector_db_factory import VectorDBFactory, get_vector_db_factory

__all__ = [
    "ChainBuilder",
    "ChatHistoryManager",
    "StreamHandler",
    "create_chat_prompts",
    "KnowledgeBaseValidator",
    "KnowledgeBaseFactory",
    "KnowledgeBaseRepository",
    "FileProcessor",
    "VectorDBManager",
    "RemoteRerankerCompressor",
    "KnowledgeManager",
    "DocumentProcessor",
    "RetrieverBuilder",
    "create_llm",
    "get_llms",
    "LLMSupplier",
    "create_embedding",
    "get_embedding",
    "EmbeddingSupplier",
    "VectorDBFactory",
    "get_vector_db_factory",
]
