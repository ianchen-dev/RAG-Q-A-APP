import os

from beanie import init_beanie
from pymongo import AsyncMongoClient

#   导入所有文档模型类
from src.models.assistant import Assistant
from src.models.chat_history import ChatHistoryMessage
from src.models.knowledgeBase import KnowledgeBase
from src.models.session import Session
from src.models.user import User


async def init_db():
    # 从环境变量获取完整的 MongoDB 连接 URL (包含认证信息和数据库名)
    mongodb_url = os.getenv("MONGODB_URL")
    if not mongodb_url:
        raise ValueError("错误: 环境变量 MONGODB_URL 未设置或为空。请检查 .env 文件。")

    # 从环境变量获取数据库名称 (或者也可以让 Beanie 从 URL 推断)
    db_name = os.getenv("MONGO_DB_NAME", "fastapi")  # 保留以防万一 URL 中未指定 DB

    print(
        f"Connecting to MongoDB using URL: {mongodb_url}"
    )  # 使用 MONGODB_URL 打印日志

    # 创建Motor客户端
    # Beanie 会自动处理 URL 中的数据库名，但显式指定也无妨
    client = AsyncMongoClient(mongodb_url)

    print("Initializing Beanie...")
    # 初始化Beanie
    # 让 Beanie 使用连接到的默认数据库，或者显式指定 client[db_name]
    await init_beanie(
        database=client[db_name],  # 使用从环境变量获取的 db_name
        document_models=[
            User,
            Session,
            Assistant,
            ChatHistoryMessage,
            KnowledgeBase,
        ],  # 添加所有文档模型类
    )
