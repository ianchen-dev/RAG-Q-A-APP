"""
批次处理器工具类
用于处理大量文档块的批次化嵌入和存储操作

"""

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# 可重试的异常类型
RETRIABLE_EXCEPTIONS = {
    # HTTP错误
    "429",  # Too Many Requests
    "502",  # Bad Gateway
    "503",  # Service Unavailable
    "504",  # Gateway Timeout
    # OpenAI特定错误
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    # 网络相关错误
    "ConnectionError",
    "TimeoutError",
    "aiohttp.ClientError",
    "httpx.TimeoutException",
    "httpx.ConnectError",
}


class RetryConfig:
    """重试配置类"""

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retriable_exceptions: Optional[Set[str]] = None,
    ):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避的底数
            jitter: 是否添加随机抖动
            retriable_exceptions: 可重试的异常类型集合
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retriable_exceptions = retriable_exceptions or RETRIABLE_EXCEPTIONS

    def calculate_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟时间

        Args:
            attempt: 当前重试次数（从0开始）

        Returns:
            延迟时间（秒）
        """
        # 计算指数退避延迟
        delay = self.initial_delay * (self.exponential_base**attempt)

        # 限制最大延迟
        delay = min(delay, self.max_delay)

        # 添加随机抖动
        if self.jitter:
            # 在±20%范围内添加随机性
            jitter_range = delay * 0.2
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter
            delay = max(0, delay)  # 确保延迟不为负

        return delay

    def is_retriable_exception(self, exception: Exception) -> bool:
        """
        判断异常是否可重试

        Args:
            exception: 异常对象

        Returns:
            是否可重试
        """
        exception_str = str(exception)
        exception_type = type(exception).__name__

        # 检查异常类型
        if exception_type in self.retriable_exceptions:
            return True

        # 检查异常消息中是否包含关键词
        for keyword in self.retriable_exceptions:
            if keyword.lower() in exception_str.lower():
                return True

        # 特殊处理HTTP状态码
        if hasattr(exception, "status_code"):
            status_code = str(exception.status_code)
            if status_code in self.retriable_exceptions:
                return True

        # 检查响应对象中的状态码
        if hasattr(exception, "response") and hasattr(
            exception.response, "status_code"
        ):
            status_code = str(exception.response.status_code)
            if status_code in self.retriable_exceptions:
                return True

        return False


def async_retry_with_exponential_backoff(
    retry_config: Optional[RetryConfig] = None,
) -> Callable:
    """
    异步函数的指数退避重试装饰器

    Args:
        retry_config: 重试配置，如果为None则使用默认配置

    Returns:
        装饰器函数
    """
    if retry_config is None:
        retry_config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    # 尝试执行函数
                    result = await func(*args, **kwargs)

                    # 如果不是第一次尝试，记录成功信息
                    if attempt > 0:
                        logger.info(
                            f"函数 {func.__name__} 在第 {attempt + 1} 次尝试后成功执行"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # 检查是否为可重试的异常
                    if not retry_config.is_retriable_exception(e):
                        logger.error(f"函数 {func.__name__} 发生不可重试异常: {e}")
                        raise e

                    # 如果已达到最大重试次数
                    if attempt >= retry_config.max_retries:
                        logger.error(
                            f"函数 {func.__name__} 达到最大重试次数 {retry_config.max_retries}，最后异常: {e}"
                        )
                        break

                    # 计算延迟时间
                    delay = retry_config.calculate_delay(attempt)

                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，"
                        f"{delay:.2f}秒后进行第 {attempt + 2} 次重试"
                    )

                    # 等待指定时间后重试
                    await asyncio.sleep(delay)

            # 所有重试都失败，抛出最后的异常
            raise last_exception

        return wrapper

    return decorator


class BatchProcessor(Generic[T]):
    """
    通用批次处理器

    支持将大量数据分批次处理，避免超出API限制
    使用信号量控制并发批次数量
    支持指数退避重试机制处理API速率限制
    """

    def __init__(
        self,
        batch_size: int = 64,
        max_concurrent_batches: int = 10,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        初始化批次处理器

        Args:
            batch_size: 每批次的大小，默认32（适配API限制）
            max_concurrent_batches: 最大并发批次数量，默认10
            retry_config: 重试配置，如果为None则使用默认配置
        """
        # 调试
        batch_size = 1024
        max_concurrent_batches = 1

        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)

        # 初始化重试配置
        self.retry_config = retry_config or RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )

        logger.info(
            f"BatchProcessor 初始化: batch_size={batch_size}, "
            f"max_concurrent_batches={max_concurrent_batches}, "
            f"max_retries={self.retry_config.max_retries}"
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
            """处理单个批次，使用信号量控制并发，支持重试机制"""
            async with self.semaphore:

                @async_retry_with_exponential_backoff(self.retry_config)
                async def process_with_retry():
                    """带重试机制的批次处理函数"""
                    logger.debug(
                        f"开始处理第 {batch_index + 1}/{len(batches)} 批次，包含 {len(batch)} 个项目"
                    )

                    # 调用处理函数
                    batch_results = await process_func(batch, *args, **kwargs)

                    return batch_results if batch_results else []

                try:
                    logger.info(
                        f"开始处理第 {batch_index + 1}/{len(batches)} 批次，包含 {len(batch)} 个项目"
                    )

                    # 使用重试机制处理批次
                    batch_results = await process_with_retry()

                    logger.info(
                        f"第 {batch_index + 1} 批次处理完成，返回 {len(batch_results) if batch_results else 0} 个结果"
                    )

                    return batch_results

                except Exception as e:
                    logger.error(
                        f"处理第 {batch_index + 1} 批次最终失败: {e}", exc_info=True
                    )
                    # 根据配置决定是否抛出异常还是返回空结果
                    if self.retry_config.is_retriable_exception(e):
                        logger.warning(
                            f"第 {batch_index + 1} 批次因API限制失败，返回空结果以继续处理其他批次"
                        )
                        return []  # 返回空结果，允许其他批次继续处理
                    else:
                        raise e  # 非API限制错误，直接抛出

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
    集成指数退避重试机制，专门处理API速率限制
    """

    def __init__(
        self,
        batch_size: int = 64,
        max_concurrent_batches: int = 5,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        初始化文档批次处理器

        Args:
            batch_size: 每批次文档数量，默认32
            max_concurrent_batches: 最大并发批次数量，默认3（更保守的设置）
            retry_config: 重试配置，如果为None则使用针对文档处理优化的默认配置
        """
        # 为文档处理设置更保守的重试配置
        if retry_config is None:
            retry_config = RetryConfig(
                max_retries=5,  # 文档处理允许更多重试
                initial_delay=2.0,  # 更长的初始延迟
                max_delay=60.0,  # 更长的最大延迟
                exponential_base=2.0,
                jitter=True,
            )

        super().__init__(batch_size, max_concurrent_batches, retry_config)
        logger.info(
            f"DocumentBatchProcessor 已初始化: batch_size={self.batch_size}, "
            f"max_concurrent_batches={self.max_concurrent_batches}, "
            f"max_retries={self.retry_config.max_retries}"
        )

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
    batch_size: int = 64,
    max_concurrent_batches: int = 10,
    retry_config: Optional[RetryConfig] = None,
) -> DocumentBatchProcessor:
    """
    获取文档批次处理器实例

    Args:
        batch_size: 批次大小，默认32
        max_concurrent_batches: 最大并发批次数量，默认3
        retry_config: 重试配置

    Returns:
        DocumentBatchProcessor实例
    """
    return DocumentBatchProcessor(
        batch_size=batch_size,
        max_concurrent_batches=max_concurrent_batches,
        retry_config=retry_config,
    )
