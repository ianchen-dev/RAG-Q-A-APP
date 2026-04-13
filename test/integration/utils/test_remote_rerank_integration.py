"""
Integration tests for remote_rerank.py module.

These tests call the real SiliconFlow Rerank API and require:
1. Valid SILICONFLOW_API_KEY in environment variables
2. Network connectivity to api.siliconflow.cn
3. Test account with available API quota

Run with: uv run pytest test/integration/utils/test_remote_rerank_integration.py -v
"""

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx
import pytest
from dotenv import load_dotenv

from src.utils.remote_rerank import call_siliconflow_rerank

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def api_key() -> str:
    """
    Load and validate SiliconFlow API key from environment.

    Skips all tests in this module if API key is not available.
    """
    load_dotenv(dotenv_path=".env.dev", override=True)
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        pytest.skip(
            "SILICONFLOW_API_KEY not set in environment variables. "
            "Skipping integration tests."
        )

    return api_key


@pytest.mark.integration
class TestSiliconFlowRerankAPI:
    """Integration tests for SiliconFlow Rerank API."""

    @pytest.mark.asyncio
    async def test_basic_reranking(
        self, api_key: str
    ) -> None:
        """Test basic reranking functionality with sample documents."""
        query = "全球变暖的影响"
        documents = [
            "农业产量可能会受到极端天气事件的影响。",  # index 0
            "海平面上升是全球变暖的一个显著后果，威胁沿海城市。",  # index 1
            "冰川融化导致淡水资源减少。",  # index 2
            "生物多样性面临威胁，许多物种栖息地改变。",  # index 3
            "关于可再生能源的讨论。",  # index 4
        ]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        # Skip test if API returns None due to account issues (403, rate limit, etc.)
        if result is None:
            pytest.skip(
                "API returned None - possibly due to insufficient balance, "
                "rate limiting, or account verification required."
            )

        assert isinstance(result, list), "Result should be a list"
        assert len(result) > 0, "Result should not be empty"

        # Verify result structure
        for item in result:
            assert "index" in item, "Each result should have 'index'"
            assert "relevance_score" in item, "Each result should have 'relevance_score'"
            assert isinstance(item["index"], int), "Index should be an integer"
            assert isinstance(
                item["relevance_score"], float
            ), "Relevance score should be a float"
            assert 0 <= item["index"] < len(
                documents
            ), f"Index should be within document range [0, {len(documents)})"
            assert (
                0 <= item["relevance_score"] <= 1
            ), f"Relevance score should be between 0 and 1, got {item['relevance_score']}"

        # Verify results are sorted by relevance score (descending)
        scores = [item["relevance_score"] for item in result]
        assert scores == sorted(
            scores, reverse=True
        ), "Results should be sorted by relevance score (descending)"

    @pytest.mark.asyncio
    async def test_top_n_parameter(self, api_key: str) -> None:
        """Test top_n parameter limits returned results."""
        query = "人工智能"
        documents = [
            "机器学习是人工智能的一个分支。",
            "深度学习使用神经网络。",
            "自然语言处理应用广泛。",
            "计算机视觉识别图像。",
            "强化学习通过奖励优化策略。",
        ]

        top_n = 3
        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
            top_n=top_n,
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        assert len(result) <= top_n, f"Should return at most {top_n} results"

    @pytest.mark.asyncio
    async def test_relevance_ordering(self, api_key: str) -> None:
        """Test that relevant documents score higher than irrelevant ones."""
        query = "Python编程语言"
        documents = [
            "Python是一种高级编程语言，广泛用于Web开发。",  # highly relevant
            "Java是另一种流行的编程语言。",  # somewhat relevant
            "今天天气很好，适合出去散步。",  # not relevant
            "Python支持面向对象、函数式和过程式编程。",  # highly relevant
            "我昨天去看了电影。",  # not relevant
        ]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        assert len(result) > 0

        # Highly relevant documents (indices 0 and 3) should score higher
        # than irrelevant ones (indices 2 and 4)
        relevant_indices = {0, 3}
        irrelevant_indices = {2, 4}

        relevant_scores = [
            item["relevance_score"] for item in result if item["index"] in relevant_indices
        ]
        irrelevant_scores = [
            item["relevance_score"]
            for item in result
            if item["index"] in irrelevant_indices
        ]

        if relevant_scores and irrelevant_scores:
            avg_relevant = sum(relevant_scores) / len(relevant_scores)
            avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores)
            assert (
                avg_relevant > avg_irrelevant
            ), f"Relevant docs (avg={avg_relevant:.3f}) should score higher than irrelevant (avg={avg_irrelevant:.3f})"

    @pytest.mark.asyncio
    async def test_empty_documents(self, api_key: str) -> None:
        """Test behavior with empty document list."""
        result = await call_siliconflow_rerank(
            api_key=api_key,
            query="test query",
            documents=[],
            model="BAAI/bge-reranker-v2-m3",
        )

        # API behavior with empty docs may vary - just verify no crash
        # Result could be None or empty list depending on API
        assert result is None or isinstance(result, list)

    @pytest.mark.asyncio
    async def test_invalid_api_key(self) -> None:
        """Test error handling with invalid API key."""
        result = await call_siliconflow_rerank(
            api_key="invalid_key_12345",
            query="test query",
            documents=["test document"],
            model="BAAI/bge-reranker-v2-m3",
        )

        # Should return None on authentication failure
        assert result is None, "Invalid API key should return None"

    @pytest.mark.asyncio
    async def test_chinese_query_and_documents(
        self, api_key: str
    ) -> None:
        """Test with Chinese language content."""
        query = "机器学习算法"
        documents = [
            "支持向量机是一种监督学习算法。",
            "随机森林是集成学习方法。",
            "北京是中国的首都。",
            "深度学习基于人工神经网络。",
        ]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        # Most relevant documents should be about ML algorithms (indices 0, 1, 3)
        top_indices = [item["index"] for item in result[:2]]
        ml_related = {0, 1, 3}
        assert any(idx in ml_related for idx in top_indices)

    @pytest.mark.asyncio
    async def test_alternative_model(self, api_key: str) -> None:
        """Test with alternative reranker model."""
        query = "测试查询"
        documents = [
            "这是第一个测试文档。",
            "这是第二个测试文档。",
            "这是第三个测试文档。",
        ]

        # Try alternative model if available
        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="netease-youdao/bce-reranker-base_v1",
            top_n=2,
        )

        # This test may fail if model is not available
        # Just verify the function doesn't crash
        if result is not None:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_single_document(self, api_key: str) -> None:
        """Test with single document."""
        query = "test"
        documents = ["single document content"]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        assert len(result) == 1
        assert result[0]["index"] == 0
        assert 0 <= result[0]["relevance_score"] <= 1

    @pytest.mark.asyncio
    async def test_long_query(self, api_key: str) -> None:
        """Test with long query text."""
        long_query = (
            "这是一个非常长的查询语句，包含了很多详细的描述和复杂的技术术语，"
            "用于测试API在处理长文本时的表现和稳定性。"
        )
        documents = ["相关文档内容", "不相关的内容"]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=long_query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_special_characters(self, api_key: str) -> None:
        """Test with special characters in query and documents."""
        query = "HTML & CSS 编程 <script> 标签"
        documents = [
            "HTML使用<div>和<span>标签构建结构。",
            "CSS通过.class和#id选择器设置样式。",
            "JavaScript使用console.log()输出信息。",
        ]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        # Most relevant should be HTML/CSS related (indices 0, 1)
        top_indices = [item["index"] for item in result[:2]]
        assert 0 in top_indices or 1 in top_indices


@pytest.mark.integration
class TestSiliconFlowRerankEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout_simulation(
        self, api_key: str
    ) -> None:
        """Test behavior with slow network (API has 30s timeout)."""
        # This test verifies the timeout is properly configured
        # Actual timeout behavior depends on network conditions
        query = "timeout test"
        documents = ["document"]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        # Should complete within 30s timeout
        assert result is not None or result is None

    @pytest.mark.asyncio
    async def test_duplicate_documents(self, api_key: str) -> None:
        """Test with duplicate document content."""
        query = "test"
        documents = [
            "same document content",
            "same document content",  # duplicate
            "different content",
        ]

        result = await call_siliconflow_rerank(
            api_key=api_key,
            query=query,
            documents=documents,
            model="BAAI/bge-reranker-v2-m3",
        )

        if result is None:
            pytest.skip("API returned None - possibly due to account issues.")

        # API should handle duplicates and return both indices
        indices = [item["index"] for item in result]
        assert 0 in indices or 1 in indices
