"""
批次处理器工具类
用于处理大量文档块的批次化嵌入和存储操作

"""

import asyncio
import logging
from typing import Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BatchProcessor(Generic[T]):
    """
    通用批次处理器

    支持将大量数据分批次处理，避免超出API限制
    使用信号量控制并发批次数量
    """

    def __init__(self, batch_size: int = 32, max_concurrent_batches: int = 10):
        """
        初始化批次处理器

        Args:
            batch_size: 每批次的大小，默认64（适配OneAPI限制）
            max_concurrent_batches: 最大并发批次数量，默认10
        """
        batch_size = 32
        max_concurrent_batches = 5
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        logger.info(
            f"BatchProcessor 初始化: batch_size={batch_size}, max_concurrent_batches={max_concurrent_batches}"
        )

    def create_batches(self, items: List[T]) -> List[List[T]]:
        """
        将列表分割成批次

        Args:
            items: 待分割的项目列表

        Returns:
            分割后的批次列表
        """
        if not items:
            return []

        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batches.append(batch)

        logger.info(
            f"将 {len(items)} 个项目分割为 {len(batches)} 个批次（每批次最多 {self.batch_size} 个）"
        )
        return batches

    async def process_batches_async(
        self, items: List[T], process_func, *args, **kwargs
    ) -> List:
        """
        异步批次处理，使用信号量控制并发

        Args:
            items: 待处理的项目列表
            process_func: 处理函数（异步函数）
            *args, **kwargs: 传递给处理函数的额外参数

        Returns:
            所有批次处理结果的合并列表
        """
        if not items:
            logger.warning("传入空列表，无需处理")
            return []

        batches = self.create_batches(items)
        logger.info(
            f"准备并发处理 {len(batches)} 个批次，最大并发数: {self.max_concurrent_batches}"
        )

        async def process_single_batch(batch_index: int, batch: List[T]):
            """处理单个批次，使用信号量控制并发"""
            async with self.semaphore:
                try:
                    logger.info(
                        f"开始处理第 {batch_index + 1}/{len(batches)} 批次，包含 {len(batch)} 个项目"
                    )

                    # 调用处理函数
                    batch_results = await process_func(batch, *args, **kwargs)

                    logger.info(
                        f"第 {batch_index + 1} 批次处理完成，返回 {len(batch_results) if batch_results else 0} 个结果"
                    )

                    return batch_results if batch_results else []

                except Exception as e:
                    logger.error(
                        f"处理第 {batch_index + 1} 批次时发生错误: {e}", exc_info=True
                    )
                    raise e

        # 创建所有批次的协程任务
        tasks = [process_single_batch(i, batch) for i, batch in enumerate(batches)]

        # 并发执行所有批次，但受信号量限制
        batch_results_list = await asyncio.gather(*tasks)

        # 合并所有批次的结果
        all_results = []
        for batch_results in batch_results_list:
            if batch_results:
                all_results.extend(batch_results)

        logger.info(
            f"所有批次处理完成，总共处理 {len(items)} 个项目，返回 {len(all_results)} 个结果"
        )
        return all_results


class DocumentBatchProcessor(BatchProcessor):
    """
    文档批次处理器
    专门用于处理文档嵌入和存储操作
    """

    def __init__(self, batch_size: int = 64, max_concurrent_batches: int = 5):
        """
        初始化文档批次处理器

        Args:
            batch_size: 每批次文档数量，默认64
            max_concurrent_batches: 最大并发批次数量，默认5
        """
        super().__init__(batch_size, max_concurrent_batches)
        logger.info("DocumentBatchProcessor 已初始化")

    async def add_documents_in_batches(
        self, collection, documents: List, batch_size: int = None
    ) -> List[str]:
        """
        分批次添加文档到集合

        Args:
            collection: 集合对象（具有aadd_documents方法）
            documents: 文档列表
            batch_size: 批次大小，如果不指定则使用默认值

        Returns:
            所有文档的ID列表
        """
        if batch_size:
            # 临时调整批次大小
            original_batch_size = self.batch_size
            self.batch_size = batch_size

        try:

            async def add_batch(doc_batch):
                """添加单个批次的文档"""
                return await collection.aadd_documents(doc_batch)

            all_ids = await self.process_batches_async(documents, add_batch)

            logger.info(
                f"成功分批次添加 {len(documents)} 个文档，获得 {len(all_ids)} 个ID"
            )
            return all_ids

        finally:
            if batch_size:
                # 恢复原始批次大小
                self.batch_size = original_batch_size


def get_batch_processor(
    batch_size: int = 64, max_concurrent_batches: int = 5
) -> DocumentBatchProcessor:
    """
    获取文档批次处理器实例

    Args:
        batch_size: 批次大小
        max_concurrent_batches: 最大并发批次数量

    Returns:
        DocumentBatchProcessor实例
    """
    return DocumentBatchProcessor(
        batch_size=batch_size, max_concurrent_batches=max_concurrent_batches
    )
