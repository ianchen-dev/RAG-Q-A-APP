"""
批次处理功能测试
测试嵌入模型批次处理功能是否正常工作
"""

import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document

from src.adapters.chroma_adapter import ChromaAdapter
from src.utils.batch_processor import BatchProcessor, DocumentBatchProcessor
from src.utils.embedding import get_embedding

# 配置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBatchProcessor:
    """测试通用批次处理器"""

    def test_create_batches(self):
        """测试批次分割功能"""
        processor = BatchProcessor(batch_size=3)

        # 测试空列表
        assert processor.create_batches([]) == []

        # 测试少于批次大小的列表
        items = [1, 2]
        batches = processor.create_batches(items)
        assert len(batches) == 1
        assert batches[0] == [1, 2]

        # 测试正好等于批次大小的列表
        items = [1, 2, 3]
        batches = processor.create_batches(items)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

        # 测试超过批次大小的列表
        items = [1, 2, 3, 4, 5, 6, 7]
        batches = processor.create_batches(items)
        assert len(batches) == 3
        assert batches[0] == [1, 2, 3]
        assert batches[1] == [4, 5, 6]
        assert batches[2] == [7]

    @pytest.mark.asyncio
    async def test_process_batches_async(self):
        """测试异步批次处理"""
        processor = BatchProcessor(batch_size=2, delay_between_batches=0.01)

        # 模拟异步处理函数
        async def mock_process_func(batch):
            await asyncio.sleep(0.001)  # 模拟处理时间
            return [f"processed_{item}" for item in batch]

        items = [1, 2, 3, 4, 5]
        results = await processor.process_batches_async(items, mock_process_func)

        expected = [
            "processed_1",
            "processed_2",
            "processed_3",
            "processed_4",
            "processed_5",
        ]
        assert results == expected


class TestDocumentBatchProcessor:
    """测试文档批次处理器"""

    def test_init(self):
        """测试文档批次处理器初始化"""
        processor = DocumentBatchProcessor(batch_size=32, delay_between_batches=0.2)
        assert processor.batch_size == 32
        assert processor.delay_between_batches == 0.2

    @pytest.mark.asyncio
    async def test_add_documents_in_batches(self):
        """测试分批次添加文档"""
        processor = DocumentBatchProcessor(batch_size=2)

        # 创建模拟文档
        documents = [
            Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(5)
        ]

        # 创建模拟集合
        mock_collection = AsyncMock()
        mock_collection.aadd_documents = AsyncMock(
            side_effect=lambda docs: [f"id_{i}" for i in range(len(docs))]
        )

        # 执行批次添加
        result_ids = await processor.add_documents_in_batches(
            mock_collection, documents
        )

        # 验证结果
        assert len(result_ids) == 5
        assert mock_collection.aadd_documents.call_count == 3  # 3个批次：2+2+1


class TestChromaAdapterBatchProcessing:
    """测试ChromaAdapter的批次处理功能"""

    @pytest.fixture
    def mock_embeddings(self):
        """创建模拟嵌入模型"""
        embeddings = Mock()
        embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]] * 10)
        return embeddings

    def test_chroma_adapter_init_with_batch_size(self, mock_embeddings):
        """测试ChromaAdapter初始化时的批次大小设置"""
        adapter = ChromaAdapter(mock_embeddings, batch_size=32)
        assert adapter.batch_processor.batch_size == 32

    @pytest.mark.asyncio
    async def test_add_documents_batch_decision(self, mock_embeddings):
        """测试添加文档时的批次处理决策"""
        adapter = ChromaAdapter(mock_embeddings, batch_size=3)

        # 模拟_get_or_create_collection方法
        mock_collection = AsyncMock()
        adapter._get_or_create_collection = Mock(return_value=mock_collection)

        # 模拟collection_exists方法
        adapter.collection_exists = AsyncMock(return_value=True)

        # 创建少量文档（不超过批次大小）
        small_docs = [
            Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(2)
        ]

        # 模拟_add_documents_direct方法
        adapter._add_documents_direct = AsyncMock(return_value=["id_0", "id_1"])

        # 执行添加
        result = await adapter.add_documents("test_collection", small_docs)

        # 验证使用了直接添加
        adapter._add_documents_direct.assert_called_once()
        assert len(result) == 2

        # 重置mock
        adapter._add_documents_direct.reset_mock()

        # 创建大量文档（超过批次大小）
        large_docs = [
            Document(page_content=f"Content {i}", metadata={"id": i}) for i in range(5)
        ]

        # 模拟_add_documents_in_batches方法
        adapter._add_documents_in_batches = AsyncMock(
            return_value=["id_0", "id_1", "id_2", "id_3", "id_4"]
        )

        # 执行添加
        result = await adapter.add_documents("test_collection", large_docs)

        # 验证使用了批次添加
        adapter._add_documents_in_batches.assert_called_once()
        assert len(result) == 5


class TestEmbeddingBatchSize:
    """测试嵌入模型批次大小配置"""

    def test_get_embedding_with_chunk_size(self):
        """测试get_embedding函数的chunk_size参数"""
        # 测试默认批次大小
        with patch("src.utils.embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            embedding = get_embedding("oneapi", "test-model", "test-key")

            # 验证OpenAIEmbeddings被正确调用
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            assert call_args[1]["chunk_size"] == 64

        # 测试自定义批次大小
        with patch("src.utils.embedding.OpenAIEmbeddings") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            embedding = get_embedding("oneapi", "test-model", "test-key", chunk_size=32)

            # 验证OpenAIEmbeddings被正确调用
            mock_openai.assert_called_once()
            call_args = mock_openai.call_args
            assert call_args[1]["chunk_size"] == 32


@pytest.mark.asyncio
async def test_integration_large_document_processing():
    """集成测试：处理大量文档的完整流程"""

    # 创建模拟嵌入模型
    mock_embeddings = Mock()
    mock_embeddings.embed_documents = Mock(
        side_effect=lambda docs: [[0.1, 0.2, 0.3]] * len(docs)
    )

    # 创建ChromaAdapter
    adapter = ChromaAdapter(mock_embeddings, batch_size=3)

    # 模拟必要的方法
    adapter._get_collection_path = Mock(return_value="/tmp/test_collection")
    adapter.collection_exists = AsyncMock(return_value=False)

    # 模拟Chroma.afrom_documents
    mock_collection = AsyncMock()
    mock_collection.aadd_documents = AsyncMock(
        side_effect=lambda docs: [f"id_{i}" for i in range(len(docs))]
    )

    with patch("src.adapters.chroma_adapter.Chroma") as mock_chroma:
        mock_chroma.afrom_documents = AsyncMock(return_value=mock_collection)

        # 直接设置_get_or_create_collection返回mock_collection
        adapter._get_or_create_collection = Mock(return_value=mock_collection)

        # 创建大量文档（模拟您的225个文档块）
        documents = [
            Document(
                page_content=f"This is content for document {i}", metadata={"doc_id": i}
            )
            for i in range(10)  # 使用10个文档进行测试
        ]

        # 执行添加操作
        result_ids = await adapter.add_documents("test_collection", documents)

        # 验证结果
        assert len(result_ids) == 10
        logger.info(f"成功处理 {len(documents)} 个文档，获得 {len(result_ids)} 个ID")


if __name__ == "__main__":
    # 运行集成测试
    asyncio.run(test_integration_large_document_processing())
    print("批次处理集成测试完成！")
