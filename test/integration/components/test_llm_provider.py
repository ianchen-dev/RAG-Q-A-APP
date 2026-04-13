"""
Integration tests for llm_provider.py module.

These tests create real LLM instances and require:
1. Valid API keys in environment variables (SILICONFLOW_API_KEY, VOLCES_API_KEY, etc.)
2. Network connectivity to LLM service endpoints
3. Test account with available API quota

Run with: uv run pytest test/integration/components/test_llm_provider.py -v
"""

import os
from typing import Any

import pytest
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from src.components.llm_provider import LLMSupplier, create_llm, get_llms


@pytest.fixture(scope="module")
def env_vars() -> dict[str, str]:
    """
    Load environment variables from .env file.

    Skips all tests in this module if required env vars are not available.
    """
    load_dotenv(dotenv_path=".env.dev", override=True)

    required_vars = {
        "SILICONFLOW_API_KEY": os.getenv("SILICONFLOW_API_KEY"),
        "VOLCES_API_KEY": os.getenv("VOLCES_API_KEY"),
        "VOLCES_MODEL": os.getenv("VOLCES_MODEL"),
    }

    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        pytest.skip(
            f"Required environment variables not set: {', '.join(missing_vars)}. "
            "Skipping integration tests."
        )

    return required_vars


@pytest.mark.integration
class TestLLMProviderSiliconFlow:
    """Integration tests for SiliconFlow LLM provider."""

    @pytest.mark.asyncio
    async def test_create_siliconflow_llm(self, env_vars: dict[str, str]) -> None:
        """Test creating SiliconFlow LLM instance."""
        llm = create_llm(
            supplier="siliconflow",
            model="deepseek-ai/DeepSeek-V3",
        )

        assert isinstance(llm, (ChatOpenAI, BaseChatOpenAI))

    @pytest.mark.asyncio
    async def test_siliconflow_invoke(self, env_vars: dict[str, str]) -> None:
        """Test invoking SiliconFlow LLM with a simple query."""
        llm = create_llm(
            supplier="siliconflow",
            model="deepseek-ai/DeepSeek-V3",
        )

        result = llm.invoke("你好")
        assert result is not None
        assert hasattr(result, "content")
        assert len(result.content) > 0


@pytest.mark.integration
class TestLLMProviderVolces:
    """Integration tests for Volces/Doubao LLM provider."""

    @pytest.mark.asyncio
    async def test_create_volces_llm(self, env_vars: dict[str, str]) -> None:
        """Test creating Volces LLM instance."""
        llm = create_llm(
            supplier="volces",
            model=env_vars["VOLCES_MODEL"],
        )

        assert isinstance(llm, (ChatOpenAI, BaseChatOpenAI))

    @pytest.mark.asyncio
    async def test_volces_invoke(self, env_vars: dict[str, str]) -> None:
        """Test invoking Volces LLM with a simple query."""
        llm = create_llm(
            supplier="volces",
            model=env_vars["VOLCES_MODEL"],
        )

        result = llm.invoke("你好")
        assert result is not None
        assert hasattr(result, "content")
        assert len(result.content) > 0


@pytest.mark.integration
class TestLLMProviderOllama:
    """Integration tests for Ollama LLM provider."""

    @pytest.mark.asyncio
    async def test_create_ollama_llm(self) -> None:
        """Test creating Ollama LLM instance."""
        # This test requires Ollama to be running locally
        llm = create_llm(
            supplier="ollama",
            model="llama2",
        )

        assert isinstance(llm, ChatOllama)

    @pytest.mark.asyncio
    async def test_ollama_invoke(self) -> None:
        """Test invoking Ollama LLM with a simple query."""
        # This test requires Ollama to be running locally
        llm = create_llm(
            supplier="ollama",
            model="llama2",
        )

        try:
            result = llm.invoke("Hello")
            assert result is not None
            assert hasattr(result, "content")
        except Exception as e:
            pytest.skip(f"Ollama not available locally: {e}")


@pytest.mark.integration
class TestLLMProviderOneAPI:
    """Integration tests for OneAPI LLM provider."""

    @pytest.mark.asyncio
    async def test_create_oneapi_llm(self) -> None:
        """Test creating OneAPI LLM instance with API key."""
        llm = create_llm(
            supplier="oneapi",
            model="deepseek",
            api_key="test_api_key_123",
        )

        assert isinstance(llm, (ChatOpenAI, BaseChatOpenAI, ChatOllama))

    @pytest.mark.asyncio
    async def test_oneapi_requires_api_key(self) -> None:
        """Test that OneAPI supplier requires API key."""
        with pytest.raises(ValueError, match="api_key is required"):
            create_llm(
                supplier="oneapi",
                model="deepseek",
            )


@pytest.mark.integration
class TestLLMProviderOpenAI:
    """Integration tests for OpenAI LLM provider."""

    @pytest.mark.asyncio
    async def test_create_openai_llm(self) -> None:
        """Test creating OpenAI LLM instance."""
        llm = create_llm(
            supplier="openai",
            model="gpt-4o",
        )

        assert isinstance(llm, ChatOpenAI)


@pytest.mark.integration
class TestLLMProviderErrors:
    """Integration tests for LLM provider error handling."""

    @pytest.mark.asyncio
    async def test_unsupported_supplier(self) -> None:
        """Test that unsupported supplier raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported supplier"):
            create_llm(
                supplier="invalid_supplier",
                model="test",
            )


@pytest.mark.integration
class TestLLMProviderLegacy:
    """Integration tests for legacy get_llms function."""

    @pytest.mark.asyncio
    async def test_get_llms_siliconflow(self, env_vars: dict[str, str]) -> None:
        """Test legacy get_llms function with SiliconFlow."""
        llm = get_llms(
            supplier="siliconflow",
            model="deepseek-ai/DeepSeek-V3",
        )

        assert isinstance(llm, (ChatOpenAI, BaseChatOpenAI))

    @pytest.mark.asyncio
    async def test_get_llms_volces(self, env_vars: dict[str, str]) -> None:
        """Test legacy get_llms function with Volces."""
        llm = get_llms(
            supplier="volces",
            model=env_vars["VOLCES_MODEL"],
        )

        assert isinstance(llm, (ChatOpenAI, BaseChatOpenAI))
