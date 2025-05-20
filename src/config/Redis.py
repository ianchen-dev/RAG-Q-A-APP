import logging
import os
from typing import Optional

import redis.asyncio as aioredis  # 使用 asyncio 版本以配合 FastAPI

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量，用于持有 Redis 客户端实例和连接池
# 使用 Optional 类型提示，因为它们在初始化之前是 None
redis_client: Optional[aioredis.Redis] = None
redis_pool: Optional[aioredis.ConnectionPool] = None


async def init_redis_pool():
    """
    根据环境变量初始化 Redis 异步连接池。
    这个函数应该在 FastAPI 应用启动时被调用。
    """
    global redis_pool, redis_client
    # 检查是否已经初始化，避免重复操作
    if redis_pool is not None:
        logger.info("Redis 连接池已初始化。")
        return

    try:
        # 从环境变量读取 Redis 配置，提供默认值
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        redis_password = os.getenv("REDIS_PASSWORD", None)

        logger.info(
            f"尝试初始化 Redis 连接池: host={redis_host}, port={redis_port}, db={redis_db}"
        )

        # 创建异步连接池
        # decode_responses=True 会让 Redis 命令返回 Python 字符串而不是字节串
        redis_pool = aioredis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            max_connections=20,  # 为连接池设置一个最大连接数
            socket_timeout=5,  # 设置连接超时时间
            socket_connect_timeout=5,  # 设置连接超时时间
        )

        # 从连接池创建一个 Redis 客户端实例，供全局使用
        redis_client = aioredis.Redis(connection_pool=redis_pool)

        # 测试连接是否成功
        await redis_client.ping()
        logger.info("Redis 连接池初始化成功，ping 测试通过。")

    except (ValueError, TypeError) as e:
        # 捕获端口或 DB 编号不是有效整数的错误
        logger.error(
            f"Redis 配置错误：请检查 .env 文件中的 REDIS_PORT 和 REDIS_DB 是否为有效的整数。错误详情: {e}"
        )
        # 设置为 None，表示初始化失败
        redis_pool = None
        redis_client = None
        # 重要：这里可以选择是否抛出异常，取决于 Redis 是否是应用启动的硬性要求
        # raise ConnectionError(f"Redis 配置无效: {e}") from e
    except aioredis.RedisError as e:
        # 捕获所有 Redis 相关的连接错误或操作错误
        logger.error(f"连接 Redis 或执行 ping 命令失败: {e}")
        redis_pool = None
        redis_client = None
        # raise ConnectionError(f"无法连接到 Redis 服务器: {e}") from e
    except Exception as e:  # 捕获其他任何预料之外的错误
        logger.error(f"初始化 Redis 时发生未知错误: {e}", exc_info=True)
        redis_pool = None
        redis_client = None
        # raise # 可以选择重新抛出，中断应用启动


async def close_redis_pool():
    """
    优雅地关闭 Redis 客户端和连接池。
    这个函数应该在 FastAPI 应用关闭时被调用。
    """
    global redis_pool, redis_client
    if redis_client:
        try:
            await redis_client.close()  # 首先关闭客户端实例
            logger.info("Redis 客户端已关闭。")
        except Exception as e:
            logger.error(f"关闭 Redis 客户端时出错: {e}")
    if redis_pool:
        try:
            await redis_pool.disconnect()  # 然后断开连接池
            logger.info("Redis 连接池已断开连接。")
        except Exception as e:
            logger.error(f"断开 Redis 连接池时出错: {e}")

    # 重置全局变量
    redis_client = None
    redis_pool = None


def get_redis_client() -> aioredis.Redis:
    """
    获取已初始化的 Redis 客户端实例。

    如果在客户端未初始化时调用，会引发 RuntimeError。
    确保在使用此函数之前，init_redis_pool 已在应用启动时成功执行。

    Returns:
        aioredis.Redis: Redis 异步客户端实例。

    Raises:
        RuntimeError: 如果 Redis 客户端尚未初始化。
    """
    if redis_client is None:
        # 这个情况理论上不应发生，除非 init_redis_pool 失败或未被调用
        logger.error("尝试获取 Redis 客户端，但它尚未初始化。")
        raise RuntimeError(
            "Redis client is not initialized. Ensure init_redis_pool() was called successfully at application startup."
        )
    return redis_client
