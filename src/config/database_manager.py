"""
数据库连接管理器单例
统一管理 MongoDB 和 Redis 连接，提供连接池管理和资源清理的统一接口

"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

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
            self._mongodb_client: Optional[AsyncIOMotorClient] = None
            self._mongodb_db: Optional[AsyncIOMotorDatabase] = None
            self._mongodb_config: Dict[str, Any] = {}

            # Redis 相关
            self._redis_client: Optional[aioredis.Redis] = None
            self._redis_pool: Optional[aioredis.ConnectionPool] = None
            self._redis_config: Dict[str, Any] = {}

            # 状态标志
            self._mongodb_initialized = False
            self._redis_initialized = False
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

            # 并行初始化 MongoDB 和 Redis
            tasks = [self._init_mongodb(), self._init_redis()]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查初始化结果
            mongodb_result, redis_result = results

            if isinstance(mongodb_result, Exception):
                logger.error(f"MongoDB 初始化失败: {mongodb_result}")
            else:
                logger.info("MongoDB 初始化成功")

            if isinstance(redis_result, Exception):
                logger.error(f"Redis 初始化失败: {redis_result}")
            else:
                logger.info("Redis 初始化成功")

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
            self._mongodb_client = AsyncIOMotorClient(
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

    async def _init_redis(self) -> None:
        """初始化 Redis 连接"""
        try:
            # 获取 Redis 配置
            self._redis_config = {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "db": int(os.getenv("REDIS_DB", "0")),
                "password": os.getenv("REDIS_PASSWORD"),
                "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
                "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
                "socket_connect_timeout": int(
                    os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")
                ),
                "retry_on_timeout": os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower()
                == "true",
                "decode_responses": True,
            }

            logger.info(
                f"连接 Redis: {self._redis_config['host']}:{self._redis_config['port']}, "
                f"数据库: {self._redis_config['db']}"
            )

            # 创建连接池
            self._redis_pool = aioredis.ConnectionPool(
                host=self._redis_config["host"],
                port=self._redis_config["port"],
                db=self._redis_config["db"],
                password=self._redis_config["password"],
                decode_responses=self._redis_config["decode_responses"],
                max_connections=self._redis_config["max_connections"],
                socket_timeout=self._redis_config["socket_timeout"],
                socket_connect_timeout=self._redis_config["socket_connect_timeout"],
                retry_on_timeout=self._redis_config["retry_on_timeout"],
            )

            # 创建 Redis 客户端
            self._redis_client = aioredis.Redis(connection_pool=self._redis_pool)

            # 测试连接
            await self._redis_client.ping()
            logger.info("Redis 连接测试成功")

            self._redis_initialized = True

        except Exception as e:
            logger.error(f"Redis 初始化失败: {e}", exc_info=True)
            # Redis 失败不应阻止应用启动，设置为 None
            self._redis_client = None
            self._redis_pool = None
            # 不抛出异常，允许应用在没有 Redis 的情况下运行

    async def get_mongodb_client(self) -> AsyncIOMotorClient:
        """获取 MongoDB 客户端实例"""
        if not self._mongodb_initialized or self._mongodb_client is None:
            await self.initialize()

        if self._mongodb_client is None:
            raise RuntimeError("MongoDB 客户端未初始化或初始化失败")

        return self._mongodb_client

    async def get_mongodb_database(self) -> AsyncIOMotorDatabase:
        """获取 MongoDB 数据库实例"""
        if not self._mongodb_initialized or self._mongodb_db is None:
            await self.initialize()

        if self._mongodb_db is None:
            raise RuntimeError("MongoDB 数据库实例未初始化或初始化失败")

        return self._mongodb_db

    async def get_redis_client(self) -> aioredis.Redis:
        """获取 Redis 客户端实例"""
        if not self._redis_initialized or self._redis_client is None:
            await self.initialize()

        if self._redis_client is None:
            raise RuntimeError(
                "Redis 客户端未初始化或初始化失败。请检查 Redis 服务是否运行正常。"
            )

        return self._redis_client

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
            "redis": {"status": "unknown", "error": None},
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

        # 检查 Redis
        try:
            if self._redis_client:
                await self._redis_client.ping()
                health_status["redis"]["status"] = "healthy"
                logger.debug("Redis 健康检查通过")
            else:
                health_status["redis"]["status"] = "not_initialized"
        except Exception as e:
            health_status["redis"]["status"] = "unhealthy"
            health_status["redis"]["error"] = str(e)
            logger.warning(f"Redis 健康检查失败: {e}")

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
            "redis": {
                "initialized": self._redis_initialized,
                "config": self._redis_config.copy() if self._redis_config else None,
            },
        }

        # 移除敏感信息
        if stats["mongodb"]["config"]:
            stats["mongodb"]["config"].pop("url", None)
        if stats["redis"]["config"]:
            stats["redis"]["config"].pop("password", None)

        # 获取 Redis 连接池信息
        if self._redis_pool:
            try:
                stats["redis"]["pool_stats"] = {
                    "created_connections": self._redis_pool.created_connections,
                    "available_connections": len(
                        self._redis_pool._available_connections
                    ),
                    "in_use_connections": len(self._redis_pool._in_use_connections),
                }
            except Exception as e:
                logger.warning(f"获取 Redis 连接池统计信息失败: {e}")

        return stats

    async def close(self) -> None:
        """关闭所有数据库连接"""
        async with self._lock:
            logger.info("开始关闭数据库连接...")

            # 并行关闭连接
            tasks = []

            if self._redis_client:
                tasks.append(self._close_redis())

            if self._mongodb_client:
                tasks.append(self._close_mongodb())

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"关闭数据库连接时发生错误: {result}")

            # 重置状态
            self._mongodb_initialized = False
            self._redis_initialized = False
            self._initialized = False

            logger.info("数据库连接关闭完成")

    async def _close_redis(self) -> None:
        """关闭 Redis 连接"""
        try:
            if self._redis_client:
                await self._redis_client.close()
                logger.info("Redis 客户端已关闭")

            if self._redis_pool:
                await self._redis_pool.disconnect()
                logger.info("Redis 连接池已断开")

        except Exception as e:
            logger.error(f"关闭 Redis 连接时出错: {e}")
        finally:
            self._redis_client = None
            self._redis_pool = None

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


async def get_redis_client() -> aioredis.Redis:
    """获取 Redis 客户端的便捷函数"""
    manager = await get_database_manager()
    return await manager.get_redis_client()
