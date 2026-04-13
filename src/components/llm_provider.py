"""LLM provider component for RAG ChatBot.

This module provides a factory function for creating language model instances
from various providers (OpenAI, SiliconFlow, Volces, Ollama, OneAPI).

The component encapsulates the complexity of initializing different LLM providers
with their specific configurations and provides a unified interface.
"""

import os
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

# Type alias for supported suppliers
LLMSupplier = Literal["openai", "siliconflow", "volces", "ollama", "oneapi"]


def create_llm(
    supplier: LLMSupplier,
    model: str,
    api_key: str | None = None,
    max_length: int = 10,
    temperature: float = 0.8,
    streaming: bool = True,
) -> ChatOpenAI | BaseChatOpenAI | ChatOllama:
    """Create a language model instance from the specified supplier.

    Args:
        supplier: The LLM provider to use. Must be one of:
            - "openai": OpenAI API
            - "siliconflow": SiliconFlow API (uses DeepSeek-V3 model)
            - "volces": Volces/Doubao API
            - "ollama": Local Ollama instance
            - "oneapi": OneAPI gateway
        model: The model name to use.
        api_key: Optional API key. Required for some suppliers (e.g., oneapi).
        max_length: Maximum tokens for response generation (default: 10).
            Note: This parameter is currently not used in most configurations.
        temperature: Sampling temperature for response generation (default: 0.8).
            Higher values (e.g., 0.8) make output more random, lower values (e.g., 0.2)
            make it more focused and deterministic.
        streaming: Whether to enable streaming responses (default: True).

    Returns:
        A language model instance compatible with LangChain.

    Raises:
        ValueError: If the supplier is not supported or model initialization fails.
        ConnectionError: If there's an error connecting to the LLM service.

    Examples:
        Create an OpenAI model:
        >>> llm = create_llm("openai", "gpt-4o", temperature=0.7)

        Create a SiliconFlow model:
        >>> llm = create_llm("siliconflow", "deepseek-ai/DeepSeek-V3")

        Create an Ollama model:
        >>> llm = create_llm("ollama", "llama2")
    """
    try:
        if supplier == "openai":
            return ChatOpenAI(model=model, temperature=temperature, streaming=streaming)

        elif supplier == "siliconflow":
            return BaseChatOpenAI(
                model="deepseek-ai/DeepSeek-V3",
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                openai_api_base="https://api.siliconflow.cn/v1",
                streaming=streaming,
            )

        elif supplier == "volces":
            return BaseChatOpenAI(
                model=os.getenv("VOLCES_MODEL"),
                openai_api_key=os.getenv("VOLCES_API_KEY"),
                openai_api_base="https://ark.cn-beijing.volces.com/api/v3/",
                stream=True,
            )

        elif supplier == "ollama":
            return ChatOllama(model=model)

        elif supplier == "oneapi":
            if api_key is None:
                raise ValueError("api_key is required for oneapi supplier")

            chat = ChatOpenAI(
                api_key=api_key,
                base_url="http://localhost:3000/v1",
                model=model,
                temperature=temperature,
                streaming=streaming,
            )
            # Validate that the chat object is of a supported type
            if not isinstance(chat, (ChatOpenAI, BaseChatOpenAI, ChatOllama)):  # type: ignore
                raise ValueError("model init error")

            return chat

        else:
            raise ValueError(f"Unsupported supplier: {supplier}")

    except Exception as e:
        print(f"Error: {e}")
        raise ConnectionError(f"Error: {e}")


# Backward compatibility alias - will be deprecated
def get_llms(
    supplier: str,
    model: str,
    api_key: str = None,
    max_length: int = 10,
    temperature: float = 0.8,
    streaming: bool = True,
) -> ChatOpenAI | BaseChatOpenAI | ChatOllama:
    """Legacy function for backward compatibility.

    .. deprecated::
        Use :func:`create_llm` instead. This function will be removed in a future version.
    """
    return create_llm(
        supplier=supplier,
        model=model,
        api_key=api_key,
        max_length=max_length,
        temperature=temperature,
        streaming=streaming,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="D:/aProject/fastapi/.env.dev")
    apikey = os.getenv("siliconflow_api_key")
    llm = create_llm(
        supplier="siliconflow",
        model="deepseek-ai/DeepSeek-V3",
        api_key=apikey,
        max_length=10086,
    )
    print(llm.invoke("你好"))
