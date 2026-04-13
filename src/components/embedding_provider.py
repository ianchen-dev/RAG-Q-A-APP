"""Embedding provider component for RAG ChatBot.

This module provides a factory function for creating embedding model instances
from various providers (Ollama, OneAPI, SiliconFlow).

The component encapsulates the complexity of initializing different embedding
providers with their specific configurations and provides a unified interface.
"""

import os
from typing import Literal

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings as OpenAIEmbeddingsType

# Type alias for supported suppliers
EmbeddingSupplier = Literal["ollama", "oneapi", "siliconflow"]

# Environment variables
ONEAPI_BASE_URL = os.getenv("ONEAPI_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")


def create_embedding(
    supplier: EmbeddingSupplier,
    model: str,
    api_key: str | None = None,
    chunk_size: int = 64,
) -> OpenAIEmbeddingsType | OllamaEmbeddings:
    """Create an embedding model instance from the specified supplier.

    Args:
        supplier: The embedding provider to use. Must be one of:
            - "ollama": Local Ollama instance
            - "oneapi": OneAPI gateway
            - "siliconflow": SiliconFlow API (uses BAAI/bge-m3 model)
        model: The model name to use.
        api_key: Optional API key. Required for oneapi supplier.
        chunk_size: Batch size for embedding requests (default: 64).
            This is optimized for OneAPI rate limits.

    Returns:
        An embedding model instance compatible with LangChain.

    Raises:
        ValueError: If the supplier is not supported or required parameters are missing.
        ConnectionError: If there's an error connecting to the embedding service.

    Examples:
        Create an Ollama embedding:
        >>> embedding = create_embedding("ollama", "nomic-embed-text")

        Create a OneAPI embedding:
        >>> embedding = create_embedding("oneapi", "text-embedding-ada-002", api_key="your-key")

        Create a SiliconFlow embedding:
        >>> embedding = create_embedding("siliconflow", "BAAI/bge-m3")
    """
    try:
        if supplier == "ollama":
            return OllamaEmbeddings(model=model)

        elif supplier == "oneapi":
            if api_key is None:
                raise ValueError("api_key is required for oneapi supplier")
            if ONEAPI_BASE_URL is None:
                raise ValueError("ONEAPI_BASE_URL environment variable is not set")

            return OpenAIEmbeddings(
                base_url=ONEAPI_BASE_URL,
                model=model,
                api_key=api_key,
                chunk_size=chunk_size,
            )

        elif supplier == "siliconflow":
            if SILICONFLOW_API_KEY is None:
                raise ValueError("SILICONFLOW_API_KEY environment variable is not set")

            return OpenAIEmbeddings(
                api_key=SILICONFLOW_API_KEY,
                base_url="https://api.siliconflow.cn",
                model="BAAI/bge-m3",
                chunk_size=chunk_size,
            )

        else:
            raise ValueError(f"Unsupported supplier: {supplier}")

    except Exception as e:
        print(f"Error: {e}")
        raise ConnectionError(f"Error: {e}")


# Backward compatibility alias - will be deprecated
def get_embedding(
    supplier: str,
    model_name: str,
    inference_api_key: str | None = None,
    chunk_size: int = 64,
) -> OpenAIEmbeddingsType | OllamaEmbeddings:
    """Legacy function for backward compatibility.

    .. deprecated::
        Use :func:`create_embedding` instead. This function will be removed in a future version.
    """
    return create_embedding(
        supplier=supplier,  # type: ignore
        model=model_name,
        api_key=inference_api_key,
        chunk_size=chunk_size,
    )
