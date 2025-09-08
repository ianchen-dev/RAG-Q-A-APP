import logging
import os

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

#   导入所有文档模型类
from src.models.assistant import Assistant
from src.models.chat_history import ChatHistoryMessage
from src.models.knowledgeBase import KnowledgeBase
from src.models.session import Session
from src.models.user import User

logger = logging.getLogger(__name__)

# 全局变量 (或更复杂的应用中的DI容器/app state) 来存储数据库实例和客户端实例
_db_instance: AsyncIOMotorDatabase = None
_client_instance: AsyncIOMotorClient = None


async def init_db():
    global _db_instance, _client_instance

    mongodb_url = os.getenv("MONGODB_URL")
    if not mongodb_url:
        logger.error("错误: 环境变量 MONGODB_URL 未设置或为空。请检查 .env 文件。")
        raise ValueError("错误: 环境变量 MONGODB_URL 未设置或为空。请检查 .env 文件。")

    db_name = os.getenv("MONGO_DB_NAME", "fastapi")
    logger.info(
        f"Connecting to MongoDB using URL: {mongodb_url} for database: {db_name}"
    )

    client = AsyncIOMotorClient(mongodb_url)

    # 存储客户端和数据库实例
    _client_instance = client
    _db_instance = client[db_name]
    logger.info(f"AsyncIOMotorClient and AsyncIOMotorDatabase instances for '{db_name}' created.")

    logger.info("Initializing Beanie...")
    await init_beanie(
        database=_db_instance,
        document_models=[
            User,
            Session,
            Assistant,
            ChatHistoryMessage,
            KnowledgeBase,
        ],
    )
    logger.info("Beanie initialization complete.")


def get_motor_db() -> AsyncIOMotorDatabase:
    """
    获取已初始化的 AsyncIOMotorDatabase 实例。
    在调用此函数之前，必须已成功执行 init_db()。
    """
    if _db_instance is None:
        logger.error("严重错误: get_motor_db() 被调用，但数据库实例尚未初始化。")
        raise RuntimeError("数据库实例尚未初始化。请确保 init_db() 已成功执行。")
    return _db_instance


def get_motor_client() -> AsyncIOMotorClient:
    """
    获取已初始化的 AsyncIOMotorClient 实例。
    在调用此函数之前，必须已成功执行 init_db()。
    """
    if _client_instance is None:
        logger.error("严重错误: get_motor_client() 被调用，但客户端实例尚未初始化。")
        raise RuntimeError("客户端实例尚未初始化。请确保 init_db() 已成功执行。")
    return _client_instance
