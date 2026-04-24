"""
数据库连接管理器单例
统一管理 MongoDB 连接，提供连接池管理和资源清理的统一接口

"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from beanie import init_beanie
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

# 导入所有文档模型类
from src.models.assistant import Assistant
from src.models.chat_history import ChatHistoryMessage
from src.models.knowledgeBase import KnowledgeBase
from src.models.session import Session
from src.models.user import User

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库连接管理器单例"""

    _instance: Optional["DatabaseManager"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化数据库管理器实例"""
        if not hasattr(self, "_initialized") or not self._initialized:
            # MongoDB 相关
            self._mongodb_client: Optional[AsyncMongoClient] = None
            self._mongodb_db: Optional[AsyncDatabase] = None
            self._mongodb_config: Dict[str, Any] = {}

            # 状态标志
            self._mongodb_initialized = False
            self._initialized = False

            # 健康检查相关
            self._last_health_check: Optional[datetime] = None
            self._health_check_interval = timedelta(minutes=5)

    async def initialize(self) -> None:
        """初始化所有数据库连接"""
        async with self._lock:
            if self._initialized:
                logger.info("数据库管理器已初始化，跳过重复初始化")
                return

            logger.info("开始初始化数据库连接管理器...")

            # 初始化 MongoDB
            await self._init_mongodb()

            self._initialized = True
            logger.info("数据库连接管理器初始化完成")

    async def _init_mongodb(self) -> None:
        """初始化 MongoDB 连接"""
        try:
            mongodb_url = os.getenv("MONGODB_URL")
            if not mongodb_url:
                raise ValueError("环境变量 MONGODB_URL 未设置或为空")

            db_name = os.getenv("MONGO_DB_NAME", "fastapi")

            # 配置连接参数
            self._mongodb_config = {
                "url": mongodb_url,
                "db_name": db_name,
                "maxPoolSize": int(os.getenv("MONGODB_MAX_POOL_SIZE", "100")),
                "minPoolSize": int(os.getenv("MONGODB_MIN_POOL_SIZE", "10")),
                "maxIdleTimeMS": int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", "30000")),
                "serverSelectionTimeoutMS": int(
                    os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")
                ),
                "connectTimeoutMS": int(
                    os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "10000")
                ),
                "socketTimeoutMS": int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "30000")),
            }

            logger.info(f"连接 MongoDB: {mongodb_url}, 数据库: {db_name}")

            # 创建客户端连接
            self._mongodb_client = AsyncMongoClient(
                mongodb_url,
                maxPoolSize=self._mongodb_config["maxPoolSize"],
                minPoolSize=self._mongodb_config["minPoolSize"],
                maxIdleTimeMS=self._mongodb_config["maxIdleTimeMS"],
                serverSelectionTimeoutMS=self._mongodb_config[
                    "serverSelectionTimeoutMS"
                ],
                connectTimeoutMS=self._mongodb_config["connectTimeoutMS"],
                socketTimeoutMS=self._mongodb_config["socketTimeoutMS"],
            )

            # 获取数据库实例
            self._mongodb_db = self._mongodb_client[db_name]

            # 测试连接
            await self._mongodb_client.admin.command("ping")
            logger.info("MongoDB 连接测试成功")

            # 初始化 Beanie ODM
            await init_beanie(
                database=self._mongodb_db,
                document_models=[
                    User,
                    Session,
                    Assistant,
                    ChatHistoryMessage,
                    KnowledgeBase,
                ],
            )
            logger.info("Beanie ODM 初始化完成")

            self._mongodb_initialized = True

        except Exception as e:
            logger.error(f"MongoDB 初始化失败: {e}", exc_info=True)
            raise

    async def get_mongodb_client(self) -> AsyncMongoClient:
        """获取 MongoDB 客户端实例"""
        if not self._mongodb_initialized or self._mongodb_client is None:
            await self.initialize()

        if self._mongodb_client is None:
            raise RuntimeError("MongoDB 客户端未初始化或初始化失败")

        return self._mongodb_client

    async def get_mongodb_database(self) -> AsyncDatabase:
        """获取 MongoDB 数据库实例"""
        if not self._mongodb_initialized or self._mongodb_db is None:
            await self.initialize()

        if self._mongodb_db is None:
            raise RuntimeError("MongoDB 数据库实例未初始化或初始化失败")

        return self._mongodb_db

    async def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        执行健康检查

        Args:
            force: 是否强制执行检查，忽略时间间隔限制

        Returns:
            包含各数据库连接状态的字典
        """
        now = datetime.now()

        # 检查是否需要执行健康检查
        if (
            not force
            and self._last_health_check is not None
            and now - self._last_health_check < self._health_check_interval
        ):
            logger.debug("跳过健康检查，距离上次检查时间过短")
            return {"skipped": True, "reason": "检查间隔未到"}

        logger.info("开始执行数据库健康检查...")

        health_status = {
            "timestamp": now.isoformat(),
            "mongodb": {"status": "unknown", "error": None},
        }

        # 检查 MongoDB
        try:
            if self._mongodb_client:
                await self._mongodb_client.admin.command("ping")
                health_status["mongodb"]["status"] = "healthy"
                logger.debug("MongoDB 健康检查通过")
            else:
                health_status["mongodb"]["status"] = "not_initialized"
        except Exception as e:
            health_status["mongodb"]["status"] = "unhealthy"
            health_status["mongodb"]["error"] = str(e)
            logger.warning(f"MongoDB 健康检查失败: {e}")

        self._last_health_check = now
        logger.info("数据库健康检查完成")

        return health_status

    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {
            "mongodb": {
                "initialized": self._mongodb_initialized,
                "config": self._mongodb_config.copy() if self._mongodb_config else None,
            },
        }

        # 移除敏感信息
        if stats["mongodb"]["config"]:
            stats["mongodb"]["config"].pop("url", None)

        return stats

    async def close(self) -> None:
        """关闭所有数据库连接"""
        async with self._lock:
            logger.info("开始关闭数据库连接...")

            # 关闭 MongoDB 连接
            if self._mongodb_client:
                await self._close_mongodb()

            # 重置状态
            self._mongodb_initialized = False
            self._initialized = False

            logger.info("数据库连接关闭完成")

    async def _close_mongodb(self) -> None:
        """关闭 MongoDB 连接"""
        try:
            if self._mongodb_client:
                self._mongodb_client.close()
                logger.info("MongoDB 客户端已关闭")
        except Exception as e:
            logger.error(f"关闭 MongoDB 连接时出错: {e}")
        finally:
            self._mongodb_client = None
            self._mongodb_db = None

    @classmethod
    async def get_instance(cls) -> "DatabaseManager":
        """获取数据库管理器实例（异步方法，自动初始化）"""
        instance = cls()
        if not instance._initialized:
            await instance.initialize()
        return instance

    @classmethod
    def get_sync_instance(cls) -> "DatabaseManager":
        """获取数据库管理器实例（同步方法，不执行初始化）"""
        return cls()


async def get_database_manager() -> DatabaseManager:
    """获取数据库管理器单例实例并确保已初始化"""
    instance = DatabaseManager()  # 这里会返回单例实例

    if not instance._initialized:
        await instance.initialize()
    return instance


# 初始化和清理函数
async def init_databases() -> None:
    """初始化所有数据库连接"""
    await get_database_manager()


async def close_databases() -> None:
    """关闭所有数据库连接"""
    # 获取单例实例（如果存在）
    if DatabaseManager._instance is not None:
        await DatabaseManager._instance.close()
        # 注意：这里不重置 _instance，因为单例对象可能被重新初始化
