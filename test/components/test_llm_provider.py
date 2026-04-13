"""Test for llm_provider component refactoring."""

import pytest
from src.components.llm_provider import create_llm, get_llms, LLMSupplier


class TestLLMProvider:
    """Test cases for LLM provider component."""

    def test_create_llm_openai(self):
        """Test creating an OpenAI LLM instance."""
        llm = create_llm(
            supplier="openai",
            model="gpt-4o",
            temperature=0.7,
            streaming=True,
        )
        assert llm is not None
        assert hasattr(llm, "model_name")

    def test_create_llm_ollama(self):
        """Test creating an Ollama LLM instance."""
        llm = create_llm(
            supplier="ollama",
            model="llama2",
        )
        assert llm is not None

    def test_create_llm_invalid_supplier(self):
        """Test that invalid supplier raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported supplier"):
            create_llm(
                supplier="invalid_supplier",
                model="test-model",
            )

    def test_create_llm_oneapi_requires_api_key(self):
        """Test that oneapi supplier requires api_key."""
        with pytest.raises(ValueError, match="api_key is required"):
            create_llm(
                supplier="oneapi",
                model="test-model",
            )

    def test_get_llms_backward_compatibility(self):
        """Test that get_llms function works for backward compatibility."""
        llm = get_llms(
            supplier="openai",
            model="gpt-4o",
            temperature=0.7,
        )
        assert llm is not None
