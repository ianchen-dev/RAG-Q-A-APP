import logging  # 导入 logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.responses import RedirectResponse

# 日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # 加载 .env 基础配置
logger.info("APP_ENV: .env.prod")
load_dotenv(
    dotenv_path=".env.prod", override=True
)  # override=True 特定环境文件覆盖 .env

# LANGCHAIN
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = (
    f"test-{datetime.now().strftime('%Y.%m.%d:%H')}"  # 自定义用例名称,使用当前日期:XX时
)

from src.config.Beanie import init_db

# --- 导入 MCP Client 管理器 ---
from src.config.mcp_client_manager import shutdown_mcp_client, startup_mcp_client
from src.config.Redis import close_redis_pool, get_redis_client, init_redis_pool
from src.models.user import User  # 导入 User 模型
from src.service.knowledgeSev import load_all_knowledge_bases_to_cache
from src.utils.pwdHash import get_password_hash  # 导入密码哈希函数


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化数据库
    logger.info("应用程序启动：正在初始化数据库...")
    await init_db()
    logger.info("数据库初始化完成。")

    # 初始化 Redis 连接池
    logger.info("应用程序启动：正在初始化 Redis 连接池...")
    await init_redis_pool()
    logger.info("Redis 连接池初始化完成。")

    # --- 启动 MCP Client ---
    logger.info("应用程序启动：正在初始化 MCP 客户端...")
    await startup_mcp_client()
    logger.info("MCP 客户端初始化完成。")
    # --- MCP Client 启动结束 ---

    # --- 修改：预加载知识库缓存 ---
    # 使用 get_redis_client() 来检查和获取客户端
    try:
        client = get_redis_client()  # 尝试获取客户端，如果未初始化会抛出 RuntimeError
        logger.info("Redis 客户端获取成功，正在预加载知识库缓存...")
        await load_all_knowledge_bases_to_cache()
        logger.info("知识库缓存预加载完成。")
    except RuntimeError as e:
        logger.warning(f"无法获取 Redis 客户端，跳过知识库缓存预加载: {e}")
    except Exception as e:  # 捕获 load_all_knowledge_bases_to_cache 可能发生的其他错误
        logger.error(f"预加载知识库缓存时发生错误: {e}", exc_info=True)
    # --- 结束修改 ---

    # --- 添加: 首次启动时创建 root 用户 ---
    try:
        # 检查 root 用户是否存在
        root_user = await User.find_one(User.username == "root")
        if not root_user:
            # 如果不存在，创建 root 用户
            hashed_password = get_password_hash("123456")
            root_user = User(
                username="root",
                password=hashed_password,
                email="root@example.com",  # 提供一个默认邮箱
            )
            await root_user.create()
            logger.info("Root user created successfully.")
        else:
            logger.info("Root user already exists.")
    except Exception as e:
        logger.error(f"Error during root user creation: {e}")
    # --- 结束添加 ---

    yield

    # 应用关闭时执行清理
    logger.info("应用程序关闭：正在关闭 Redis 连接池...")
    await close_redis_pool()
    logger.info("Redis 连接池已关闭。")

    # --- 关闭 MCP Client ---
    logger.info("应用程序关闭：正在关闭 MCP 客户端...")
    await shutdown_mcp_client()
    logger.info("MCP 客户端已关闭。")
    # --- MCP Client 关闭结束 ---

    logger.info("应用程序已成功关闭。")


app = FastAPI(lifespan=lifespan)

# 中间件 src.middleware  ----------------------------------------------
# cors
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 打印请求信息
from src.middleware.reqInfo import request_info_middleware

app.middleware("http")(request_info_middleware)

# --- 添加：响应时间中间件 ---
from src.middleware.resTime import add_process_time_header

app.middleware("http")(add_process_time_header)
# --- 结束添加 ---

#  import router -------------------------------------------------------
from src.router.agentRouter import AgentRouter
from src.router.assistantRouter import AssistantRouter
from src.router.auth import AuthRouter
from src.router.chatRouter import ChatRouter
from src.router.knowledgeRouter import knowledgeRouter
from src.router.sessionRouter import SessionRouter
from src.router.userRouter import UserRouter

app.include_router(router=AuthRouter)
app.include_router(router=UserRouter, prefix="/user", tags=["user"])
app.include_router(router=ChatRouter, prefix="/chat", tags=["chat"])
app.include_router(router=knowledgeRouter, prefix="/knowledge", tags=["knowledge"])
app.include_router(router=SessionRouter, prefix="/session", tags=["session"])
app.include_router(router=AssistantRouter, prefix="/assistant", tags=["assistant"])
app.include_router(router=AgentRouter, prefix="/agent", tags=["agent"])


# 当访问路径为/ ，重定向路由到/docs
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


#  静态文件  -----------------------------------------------------------
# from fastapi.staticfiles import StaticFiles

# app.mount("/static", StaticFiles(directory="static"), name="static")


# 启动web服务 ----------------------------------------------------------

# # 如果启用 MCP
# import sys
# import asyncio


import uvicorn

# # --- 自定义服务器类 解决Windows上 FastAPI/Asyncio 子进程 `NotImplementedError`
# class ProactorServer(uvicorn.Server):
#     def run(self, sockets=None):
#         # 在服务器运行前设置事件循环策略 (仅 Windows)
#         if sys.platform == "win32":
#             print("Setting ProactorEventLoopPolicy for Uvicorn server on Windows.")
#             asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
#         # 使用 asyncio.run 启动服务器的 serve 方法
#         asyncio.run(self.serve(sockets=sockets))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # 默认端口设置为8080
    host = os.getenv("HOST", "127.0.0.1")  # 默认主机设置为127.0.0.1
    uvicorn.run("main:app", host=host, port=port)
    # uvicorn.run("main:app", host=host, port=port, reload=True)

    # # 如果启用 MCP
    # # --- 修改服务器启动方式 ---
    # print(f"Starting MCP Agent server with ProactorServer at http://{host}:{port}")

    # # 1. 创建 Uvicorn 配置，确保 reload=False
    # config = uvicorn.Config(app="main:app", host=host, port=port, reload=False)

    # # 2. 实例化自定义服务器
    # server = ProactorServer(config=config)
    # server.run()
