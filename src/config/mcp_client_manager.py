import asyncio
import logging
import os
import random
from typing import Any, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)


# MCP重试配置
class MCPRetryConfig:
    """MCP客户端重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: float = 0.1,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟时间（指数退避 + 随机抖动）"""
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)

        if self.jitter:
            # 添加随机抖动，范围为 ±jitter_range%
            jitter_amount = delay * self.jitter_range * (random.random() * 2 - 1)
            delay = max(0, delay + jitter_amount)

        return delay


def load_mcp_retry_config() -> MCPRetryConfig:
    """
    从环境变量加载MCP重试配置

    支持的环境变量:
    - MCP_RETRY_MAX_RETRIES: 最大重试次数 (默认: 3)
    - MCP_RETRY_BASE_DELAY: 基础延迟时间，秒 (默认: 1.0)
    - MCP_RETRY_MAX_DELAY: 最大延迟时间，秒 (默认: 30.0)
    - MCP_RETRY_EXPONENTIAL_BASE: 指数退避底数 (默认: 2.0)
    - MCP_RETRY_ENABLE_JITTER: 是否启用随机抖动 (默认: true)
    - MCP_RETRY_JITTER_RANGE: 抖动范围比例 (默认: 0.2)
    """
    return MCPRetryConfig(
        max_retries=int(os.getenv("MCP_RETRY_MAX_RETRIES", "3")),
        base_delay=float(os.getenv("MCP_RETRY_BASE_DELAY", "1.0")),
        max_delay=float(os.getenv("MCP_RETRY_MAX_DELAY", "30.0")),
        exponential_base=float(os.getenv("MCP_RETRY_EXPONENTIAL_BASE", "2.0")),
        jitter=os.getenv("MCP_RETRY_ENABLE_JITTER", "true").lower() == "true",
        jitter_range=float(os.getenv("MCP_RETRY_JITTER_RANGE", "0.2")),
    )


# 默认重试配置（从环境变量加载）
DEFAULT_RETRY_CONFIG = load_mcp_retry_config()

# 记录加载的配置
logger.info(
    f"MCP重试配置已加载: "
    f"最大重试={DEFAULT_RETRY_CONFIG.max_retries}次, "
    f"基础延迟={DEFAULT_RETRY_CONFIG.base_delay}s, "
    f"最大延迟={DEFAULT_RETRY_CONFIG.max_delay}s, "
    f"指数底数={DEFAULT_RETRY_CONFIG.exponential_base}, "
    f"抖动={'启用' if DEFAULT_RETRY_CONFIG.jitter else '禁用'}"
)


def is_retryable_error(error: Exception) -> bool:
    """判断错误是否可以重试"""
    # 定义可重试的错误类型
    retryable_errors = (
        ConnectionError,
        TimeoutError,
        OSError,
        McpError,
    )

    # 检查错误类型
    if isinstance(error, retryable_errors):
        return True

    # 检查错误消息中的关键词
    error_message = str(error).lower()
    retryable_keywords = [
        "connection",
        "timeout",
        "network",
        "refused",
        "reset",
        "closed",
        "unavailable",
        "temporary",
    ]

    return any(keyword in error_message for keyword in retryable_keywords)


async def retry_with_backoff(
    func, retry_config: MCPRetryConfig = DEFAULT_RETRY_CONFIG, context: str = "操作"
):
    """
    带指数退避的异步重试装饰器

    Args:
        func: 需要重试的异步函数
        retry_config: 重试配置
        context: 操作上下文描述，用于日志

    Returns:
        函数执行结果

    Raises:
        最后一次重试的异常
    """
    last_exception = None

    for attempt in range(retry_config.max_retries + 1):  # +1 包括首次尝试
        try:
            logger.info(
                f"尝试{context} (第{attempt + 1}次，共{retry_config.max_retries + 1}次)"
            )
            result = await func()

            if attempt > 0:
                logger.info(f"✅ {context}在第{attempt + 1}次尝试中成功")

            return result

        except Exception as e:
            last_exception = e

            # 检查是否可以重试
            if not is_retryable_error(e):
                logger.error(f"❌ {context}失败：不可重试的错误 - {e}")
                raise e

            # 如果这是最后一次尝试，不再重试
            if attempt >= retry_config.max_retries:
                logger.error(
                    f"❌ {context}在{retry_config.max_retries + 1}次尝试后仍然失败"
                )
                break

            # 计算延迟时间
            delay = retry_config.get_delay(attempt)
            logger.warning(
                f"⚠️  {context}第{attempt + 1}次尝试失败: {e}, "
                f"将在{delay:.1f}秒后重试..."
            )

            # 等待后重试
            await asyncio.sleep(delay)

    # 抛出最后一次的异常
    if last_exception:
        raise last_exception


# MCP服务配置
MCP_CONFIG = {
    "howtocook-mcp": {
        "command": "npx",
        "args": ["-y", "howtocook-mcp"],
        "transport": "stdio",  # 添加transport字段，stdio用于命令行工具
    },
    "amap-maps": {
        "command": "npx",
        "args": ["-y", "@amap/amap-maps-mcp-server"],
        "env": {"AMAP_MAPS_API_KEY": "d0ba669a99119b0245cdc0f26cc45607"},
        "transport": "stdio",  # 添加transport字段，stdio用于命令行工具
    },
    # "current_datetime": {
    #     "url": "http://127.0.0.1:8080/mcp",
    #     "transport": "streamable_http"  # HTTP传输用于远程服务
    # },
    # 可以添加更多MCP服务配置
}


class ApplicationMCPClient:
    """
    Manages the lifecycle of the MultiServerMCPClient at an application level.
    支持重试机制的MCP客户端管理器
    """

    _instance: Optional[MultiServerMCPClient] = None
    _tools: List[Any] = []  # Cache for tools
    _retry_config: MCPRetryConfig = DEFAULT_RETRY_CONFIG

    @classmethod
    def set_retry_config(cls, retry_config: MCPRetryConfig):
        """设置重试配置"""
        cls._retry_config = retry_config
        logger.info(f"MCP重试配置已更新: 最大重试{retry_config.max_retries}次")

    async def _init_mcp_client(self) -> MultiServerMCPClient:
        """初始化MCP客户端实例（内部方法，用于重试）"""
        logger.debug("正在创建MultiServerMCPClient实例...")
        return MultiServerMCPClient(MCP_CONFIG)

    async def _get_mcp_tools(self) -> List[Any]:
        """获取MCP工具列表（内部方法，用于重试）"""
        if ApplicationMCPClient._instance is None:
            raise RuntimeError("MCP客户端实例不存在，无法获取工具")

        logger.debug("正在获取MCP工具列表...")
        tools = await ApplicationMCPClient._instance.get_tools()
        logger.debug(f"成功获取{len(tools)}个MCP工具")
        return tools

    async def startup(self, enable_retry: bool = True):
        """
        Initializes the MultiServerMCPClient and fetches its tools.
        Should be called during application startup.

        Args:
            enable_retry: 是否启用重试机制，默认为True
        """
        if ApplicationMCPClient._instance is not None:
            logger.info("MultiServerMCPClient already initialized.")
            return

        logger.info("正在初始化 MCP 客户端...")

        try:
            if enable_retry:
                # 使用重试机制初始化客户端
                ApplicationMCPClient._instance = await retry_with_backoff(
                    self._init_mcp_client, self._retry_config, "MCP客户端初始化"
                )

                # 使用重试机制获取工具
                ApplicationMCPClient._tools = await retry_with_backoff(
                    self._get_mcp_tools, self._retry_config, "MCP工具获取"
                )
            else:
                # 不使用重试机制（用于测试或特殊场景）
                logger.info("跳过重试机制，直接初始化MCP客户端...")
                ApplicationMCPClient._instance = await self._init_mcp_client()
                ApplicationMCPClient._tools = await self._get_mcp_tools()

            logger.info(
                f"✅ MCP客户端初始化成功，已加载 {len(ApplicationMCPClient._tools)} 个工具"
            )

            # 记录加载的工具详情
            for i, tool in enumerate(ApplicationMCPClient._tools, 1):
                logger.info(f"  {i}. MCP工具: {tool.name}")

        except Exception as e:
            logger.error(
                f"❌ MCP客户端初始化最终失败: {e}",
                exc_info=True,
            )
            # 清理失败的实例
            ApplicationMCPClient._instance = None
            ApplicationMCPClient._tools = []

            # 记录优雅降级信息
            logger.warning(
                "⚠️  MCP客户端不可用，应用将在不使用MCP工具的情况下继续运行。"
                "这可能会影响某些功能，但不会阻止应用启动。"
            )

    async def shutdown(self):
        """
        Shuts down the MultiServerMCPClient.
        Should be called during application shutdown.
        """
        if ApplicationMCPClient._instance is not None:
            logger.info("Shutting down MultiServerMCPClient...")
            try:
                # 新版本的MultiServerMCPClient可能需要显式关闭
                # 检查是否有close方法
                if hasattr(ApplicationMCPClient._instance, "close"):
                    await ApplicationMCPClient._instance.close()
                logger.info("MultiServerMCPClient shut down successfully.")
            except Exception as e:
                logger.error(
                    f"Error shutting down MultiServerMCPClient: {e}", exc_info=True
                )
            finally:
                ApplicationMCPClient._instance = None
                ApplicationMCPClient._tools = []
        else:
            logger.info("MultiServerMCPClient was not running or already shut down.")

    async def get_mcp_tools(self, enable_retry: bool = True) -> List[Any]:
        """
        Returns the cached MCP tools.

        Args:
            enable_retry: 是否在重新获取工具时启用重试机制
        """
        # 如果工具缓存为空但客户端实例存在，尝试重新获取
        if not ApplicationMCPClient._tools and ApplicationMCPClient._instance:
            logger.warning("MCP工具列表为空但客户端实例存在，尝试重新获取工具...")
            try:
                if enable_retry:
                    # 使用重试机制重新获取工具
                    ApplicationMCPClient._tools = await retry_with_backoff(
                        self._get_mcp_tools, self._retry_config, "MCP工具重新获取"
                    )
                else:
                    # 直接重新获取工具
                    ApplicationMCPClient._tools = await self._get_mcp_tools()

                logger.info(
                    f"✅ 重新获取了 {len(ApplicationMCPClient._tools)} 个MCP工具"
                )

            except Exception as e:
                logger.error(f"❌ 重新获取MCP工具失败: {e}")
                return []

        elif not ApplicationMCPClient._instance:
            logger.warning(
                "⚠️  MCP客户端未初始化，请先调用startup()方法。返回空工具列表。"
            )
            return []

        return ApplicationMCPClient._tools

    async def health_check(self) -> bool:
        """
        检查MCP客户端健康状态

        Returns:
            bool: True表示健康，False表示不健康
        """
        try:
            if ApplicationMCPClient._instance is None:
                logger.debug("MCP客户端实例不存在")
                return False

            # 尝试获取工具列表作为健康检查
            tools = await ApplicationMCPClient._instance.get_tools()
            logger.debug(f"MCP健康检查通过，当前有{len(tools)}个工具可用")
            return True

        except Exception as e:
            logger.warning(f"MCP健康检查失败: {e}")
            return False

    async def get_client_status(self) -> dict:
        """
        获取MCP客户端状态信息

        Returns:
            dict: 包含客户端状态的字典
        """
        return {
            "initialized": ApplicationMCPClient._instance is not None,
            "tools_count": len(ApplicationMCPClient._tools),
            "tools_available": len(ApplicationMCPClient._tools) > 0,
            "retry_config": {
                "max_retries": self._retry_config.max_retries,
                "base_delay": self._retry_config.base_delay,
                "max_delay": self._retry_config.max_delay,
            },
        }


# Singleton instance of the manager
app_mcp_client_manager = ApplicationMCPClient()


# Public functions to interact with the manager
async def startup_mcp_client(enable_retry: bool = True):
    """
    启动MCP客户端

    Args:
        enable_retry: 是否启用重试机制，默认为True
    """
    await app_mcp_client_manager.startup(enable_retry=enable_retry)


async def shutdown_mcp_client():
    """关闭MCP客户端"""
    await app_mcp_client_manager.shutdown()


async def get_cached_mcp_tools(enable_retry: bool = True) -> List[Any]:
    """
    获取缓存的MCP工具列表

    Args:
        enable_retry: 是否在重新获取工具时启用重试机制

    Returns:
        List[Any]: MCP工具列表
    """
    return await app_mcp_client_manager.get_mcp_tools(enable_retry=enable_retry)


async def mcp_health_check() -> bool:
    """
    检查MCP客户端健康状态

    Returns:
        bool: True表示健康，False表示不健康
    """
    return await app_mcp_client_manager.health_check()


async def get_mcp_client_status() -> dict:
    """
    获取MCP客户端状态信息

    Returns:
        dict: 包含客户端状态的字典
    """
    return await app_mcp_client_manager.get_client_status()


def set_mcp_retry_config(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    jitter_range: float = 0.2,
):
    """
    设置MCP重试配置

    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数退避的底数
        jitter: 是否启用随机抖动
        jitter_range: 抖动范围（比例）
    """
    retry_config = MCPRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        jitter_range=jitter_range,
    )
    ApplicationMCPClient.set_retry_config(retry_config)
