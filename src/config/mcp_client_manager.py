import logging
from typing import Any, List, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# MCP服务配置
MCP_CONFIG = {
    "howtocook-mcp": {
        "command": "npx", 
        "args": ["-y", "howtocook-mcp"],
        "transport": "stdio"  # 添加transport字段，stdio用于命令行工具
    },
    "amap-maps": {
        "command": "npx",
        "args": ["-y", "@amap/amap-maps-mcp-server"],
        "env": {"AMAP_MAPS_API_KEY": "d0ba669a99119b0245cdc0f26cc45607"},
        "transport": "stdio"  # 添加transport字段，stdio用于命令行工具
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
    """

    _instance: Optional[MultiServerMCPClient] = None
    _tools: List[Any] = []  # Cache for tools

    async def startup(self):
        """
        Initializes the MultiServerMCPClient and fetches its tools.
        Should be called during application startup.
        """
        if ApplicationMCPClient._instance is None:
            logger.info(
                "Initializing MultiServerMCPClient for application lifecycle..."
            )
            try:
                ApplicationMCPClient._instance = MultiServerMCPClient(MCP_CONFIG)
                # 使用新的API方式获取工具，不再使用异步上下文管理器
                ApplicationMCPClient._tools = await ApplicationMCPClient._instance.get_tools()
                logger.info(
                    f"MultiServerMCPClient started and {len(ApplicationMCPClient._tools)} tools fetched."
                )
                for tool in ApplicationMCPClient._tools:
                    logger.debug(f"  MCP Tool Loaded: {tool.name}")
            except Exception as e:
                logger.error(
                    f"Failed to start MultiServerMCPClient or fetch tools: {e}",
                    exc_info=True,
                )
                # 清理失败的实例
                ApplicationMCPClient._instance = None
                ApplicationMCPClient._tools = []  # Ensure tools list is empty on failure
        else:
            logger.info("MultiServerMCPClient already initialized.")

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
                if hasattr(ApplicationMCPClient._instance, 'close'):
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

    async def get_mcp_tools(self) -> List[Any]:
        """
        Returns the cached MCP tools.
        """
        if not ApplicationMCPClient._tools and ApplicationMCPClient._instance:
            # This case might occur if startup was successful but somehow tools array is empty,
            # or if called before startup completes fully.
            logger.warning(
                "MCP tools list is empty but client instance exists. Attempting to re-fetch."
            )
            try:
                ApplicationMCPClient._tools = await ApplicationMCPClient._instance.get_tools()
                logger.info(f"Re-fetched {len(ApplicationMCPClient._tools)} MCP tools.")
            except Exception as e:
                logger.error(f"Failed to re-fetch MCP tools: {e}")
                return []
        elif not ApplicationMCPClient._instance:
            logger.warning(
                "MCP client not initialized. Call startup() first. Returning empty tools list."
            )
            return []
        return ApplicationMCPClient._tools


# Singleton instance of the manager
app_mcp_client_manager = ApplicationMCPClient()


# Public functions to interact with the manager
async def startup_mcp_client():
    await app_mcp_client_manager.startup()


async def shutdown_mcp_client():
    await app_mcp_client_manager.shutdown()


async def get_cached_mcp_tools() -> List[Any]:
    return await app_mcp_client_manager.get_mcp_tools()
