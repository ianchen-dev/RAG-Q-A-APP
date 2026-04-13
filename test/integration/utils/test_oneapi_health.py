"""
Integration tests for oneapi_health.py module.

These tests check the health of OneAPI service and require:
1. Valid ONEAPI_BASE_URL and ONEAPI_API_KEY in environment variables
2. Network connectivity to OneAPI gateway
3. OneAPI service to be running

Run with: uv run pytest test/integration/utils/test_oneapi_health.py -v
"""

import asyncio
import os
from typing import Any

import pytest
from dotenv import load_dotenv

from src.utils.oneapi_health import (
    check_oneapi_health,
    check_oneapi_models,
)


@pytest.fixture(scope="module")
def oneapi_config() -> dict[str, str]:
    """
    Load OneAPI configuration from environment.

    Skips all tests in this module if required env vars are not available.
    """
    load_dotenv(dotenv_path=".env.dev", override=True)

    base_url = os.getenv("ONEAPI_BASE_URL")
    api_key = os.getenv("ONEAPI_API_KEY")

    if not base_url or not api_key:
        pytest.skip(
            "ONEAPI_BASE_URL and ONEAPI_API_KEY not set in environment variables. "
            "Skipping integration tests."
        )

    return {
        "base_url": base_url,
        "api_key": api_key,
    }


@pytest.mark.integration
class TestOneAPIHealth:
    """Integration tests for OneAPI health check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, oneapi_config: dict[str, str]) -> None:
        """Test successful health check."""
        result = await check_oneapi_health(
            base_url=oneapi_config["base_url"],
            api_key=oneapi_config["api_key"],
        )

        # Result should contain status information
        assert result is not None
        assert "status" in result

    @pytest.mark.asyncio
    async def test_health_check_with_embeddings(
        self, oneapi_config: dict[str, str]
    ) -> None:
        """Test health check including embedding model."""
        result = await check_oneapi_health(
            base_url=oneapi_config["base_url"],
            api_key=oneapi_config["api_key"],
            check_embeddings=True,
            embedding_model="BAAI/bge-large-zh-v1.5",
        )

        assert result is not None
        assert "status" in result

    @pytest.mark.asyncio
    async def test_health_check_invalid_api_key(self, oneapi_config: dict[str, str]) -> None:
        """Test health check with invalid API key."""
        result = await check_oneapi_health(
            base_url=oneapi_config["base_url"],
            api_key="invalid_api_key_12345",
        )

        # Should return unhealthy status
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_invalid_url(self) -> None:
        """Test health check with invalid URL."""
        result = await check_oneapi_health(
            base_url="http://invalid-url-12345:3000",
            api_key="test_key",
        )

        # Should return unhealthy status due to connection error
        assert result["status"] == "unhealthy"


@pytest.mark.integration
class TestOneAPIModels:
    """Integration tests for OneAPI model checking."""

    @pytest.mark.asyncio
    async def test_check_models_success(self, oneapi_config: dict[str, str]) -> None:
        """Test checking available models."""
        result = await check_oneapi_models(
            base_url=oneapi_config["base_url"],
            api_key=oneapi_config["api_key"],
        )

        assert result is not None
        assert "status" in result

        if result["status"] == "healthy":
            assert "models" in result
            assert isinstance(result["models"], list)

    @pytest.mark.asyncio
    async def test_check_models_invalid_api_key(self, oneapi_config: dict[str, str]) -> None:
        """Test checking models with invalid API key."""
        result = await check_oneapi_models(
            base_url=oneapi_config["base_url"],
            api_key="invalid_api_key_12345",
        )

        # Should return unhealthy status
        assert result["status"] == "unhealthy"
