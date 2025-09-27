"""
信号量并发控制批次处理功能测试
测试使用信号量限制并发批次数量的功能
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

from src.adapters.chroma_adapter import ChromaAdapter
from src.utils.batch_processor import BatchProcessor, DocumentBatchProcessor

# 配置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSemaphoreBatchProcessor:
    """测试信号量控制的批次处理器"""

    @pytest.mark.asyncio
    async def test_semaphore_concurrent_control(self):
        """测试信号量并发控制"""
        max_concurrent = 2
        processor = BatchProcessor(batch_size=1, max_concurrent_batches=max_concurrent)

        # 记录同时运行的任务数量
        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_process_func(batch):
            nonlocal concurrent_count, max_concurrent_observed

            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)

            # 模拟处理时间
            await asyncio.sleep(0.1)

            concurrent_count -= 1
            return [f"processed_{batch[0]}"]

        # 创建足够多的项目来测试并发控制
        items = list(range(6))  # 6个项目，每个项目一个批次

        start_time = time.time()
        results = await processor.process_batches_async(items, mock_process_func)
        end_time = time.time()

        # 验证结果
        assert len(results) == 6
        expected = [f"processed_{i}" for i in range(6)]
        assert results == expected

        # 验证并发控制：最大并发数不应超过设定值
        assert max_concurrent_observed <= max_concurrent

        # 验证处理时间：由于并发限制，应该需要至少3个时间段（6个任务/2并发）
        expected_min_time = 0.3  # 3个时间段 * 0.1秒
        assert end_time - start_time >= expected_min_time

        logger.info(
            f"最大并发观察值: {max_concurrent_observed}, 处理时间: {end_time - start_time:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_semaphore_error_handling(self):
        """测试信号量控制下的错误处理"""
        processor = BatchProcessor(batch_size=1, max_concurrent_batches=2)

        call_count = 0

        async def failing_process_func(batch):
            nonlocal call_count
            call_count += 1

            if call_count == 2:  # 第二个批次失败
                raise ValueError("模拟处理失败")

            await asyncio.sleep(0.05)
            return [f"processed_{batch[0]}"]

        items = list(range(4))

        # 应该在第二个批次时抛出异常
        with pytest.raises(ValueError, match="模拟处理失败"):
            await processor.process_batches_async(items, failing_process_func)

    def test_semaphore_initialization(self):
        """测试信号量初始化"""
        processor = BatchProcessor(batch_size=32, max_concurrent_batches=3)

        assert processor.batch_size == 32
        assert processor.max_concurrent_batches == 3
        assert processor.semaphore._value == 3  # 信号量初始值


class TestDocumentBatchProcessorSemaphore:
    """测试文档批次处理器的信号量功能"""

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self):
        """测试并发文档处理"""
        processor = DocumentBatchProcessor(batch_size=2, max_concurrent_batches=2)

        # 创建模拟文档
        documents = [
            Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(6)
        ]

        # 跟踪并发调用
        active_calls = 0
        max_active = 0

        async def mock_add_documents(doc_batch):
            nonlocal active_calls, max_active

            active_calls += 1
            max_active = max(max_active, active_calls)

            # 模拟API调用时间
            await asyncio.sleep(0.1)

            active_calls -= 1
            return [f"id_{i}" for i in range(len(doc_batch))]

        # 创建模拟集合
        mock_collection = AsyncMock()
        mock_collection.aadd_documents = mock_add_documents

        start_time = time.time()
        result_ids = await processor.add_documents_in_batches(
            mock_collection, documents
        )
        end_time = time.time()

        # 验证结果
        assert len(result_ids) == 6  # 3个批次，每批次2个文档，每个文档返回1个ID

        # 验证并发控制
        assert max_active <= 2  # 不应超过最大并发数

        # 验证处理时间（3个批次，2个并发，应该需要2个时间段）
        expected_min_time = 0.2  # 2个时间段 * 0.1秒
        assert end_time - start_time >= expected_min_time

        logger.info(
            f"文档处理 - 最大并发: {max_active}, 处理时间: {end_time - start_time:.2f}s"
        )


class TestChromaAdapterSemaphore:
    """测试ChromaAdapter的信号量并发控制"""

    @pytest.fixture
    def mock_embeddings(self):
        """创建模拟嵌入模型"""
        embeddings = Mock()
        embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 10)
        return embeddings

    def test_chroma_adapter_semaphore_init(self, mock_embeddings):
        """测试ChromaAdapter的信号量初始化"""
        adapter = ChromaAdapter(
            mock_embeddings, batch_size=32, max_concurrent_batches=3
        )

        assert adapter.batch_processor.batch_size == 32
        assert adapter.batch_processor.max_concurrent_batches == 3
        assert adapter.batch_processor.semaphore._value == 3

    @pytest.mark.asyncio
    async def test_chroma_concurrent_batch_processing(self, mock_embeddings):
        """测试ChromaAdapter的并发批次处理"""
        adapter = ChromaAdapter(mock_embeddings, batch_size=2, max_concurrent_batches=2)

        # 模拟集合和方法
        mock_collection = AsyncMock()
        adapter._get_or_create_collection = Mock(return_value=mock_collection)
        adapter.collection_exists = AsyncMock(return_value=True)

        # 跟踪并发调用
        active_calls = 0
        max_active = 0

        async def mock_add_documents(docs):
            nonlocal active_calls, max_active

            active_calls += 1
            max_active = max(max_active, active_calls)

            await asyncio.sleep(0.05)  # 模拟API调用

            active_calls -= 1
            return [f"id_{i}" for i in range(len(docs))]

        mock_collection.aadd_documents = mock_add_documents

        # 创建大量文档
        documents = [
            Document(page_content=f"Content {i}", metadata={"id": i})
            for i in range(8)  # 8个文档，批次大小为2，将产生4个批次
        ]

        start_time = time.time()
        result_ids = await adapter.add_documents("test_collection", documents)
        end_time = time.time()

        # 验证结果
        assert len(result_ids) == 8

        # 验证并发控制
        assert max_active <= 2  # 不应超过最大并发数

        logger.info(
            f"ChromaAdapter - 最大并发: {max_active}, 处理时间: {end_time - start_time:.2f}s"
        )


@pytest.mark.asyncio
async def test_performance_comparison():
    """性能对比测试：信号量并发 vs 顺序处理"""

    # 模拟处理函数
    async def mock_process(batch):
        await asyncio.sleep(0.05)  # 模拟API调用时间
        return [f"result_{item}" for item in batch]

    items = list(range(10))

    # 测试顺序处理（并发数为1）
    sequential_processor = BatchProcessor(batch_size=2, max_concurrent_batches=1)
    start_time = time.time()
    sequential_results = await sequential_processor.process_batches_async(
        items, mock_process
    )
    sequential_time = time.time() - start_time

    # 测试并发处理（并发数为3）
    concurrent_processor = BatchProcessor(batch_size=2, max_concurrent_batches=3)
    start_time = time.time()
    concurrent_results = await concurrent_processor.process_batches_async(
        items, mock_process
    )
    concurrent_time = time.time() - start_time

    # 验证结果一致性
    assert sequential_results == concurrent_results

    # 验证并发处理更快
    assert concurrent_time < sequential_time

    speedup = sequential_time / concurrent_time
    logger.info(
        f"性能提升: {speedup:.2f}x (顺序: {sequential_time:.2f}s, 并发: {concurrent_time:.2f}s)"
    )

    # 并发处理应该有显著的性能提升
    assert speedup > 1.5


if __name__ == "__main__":
    # 运行性能对比测试
    asyncio.run(test_performance_comparison())
    print("信号量并发控制测试完成！")
