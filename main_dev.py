import asyncio
import sys

# window本地开发设置适配的事件循环
if sys.platform == "win32":
    # Set the event loop policy for Windows right at the start.
    # This ensures that any subsequent asyncio operations, including
    # Uvicorn's server setup and FastAPI's lifespan events,
    # will use the ProactorEventLoop if a new loop is created by asyncio.run().
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("Successfully set WindowsProactorEventLoopPolicy at script startup.")
    except Exception as e:
        print(f"Error setting WindowsProactorEventLoopPolicy at startup: {e}")

import os
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from src.config.logging_config import setup_development_logging
from src.config.mcp_client_manager import shutdown_mcp_client, startup_mcp_client

# 初始化日志系统
logger = setup_development_logging(log_dir="./log")

load_dotenv()  # 加载 .env 基础配置
logger.info("APP_ENV: .env")
# app_env = os.getenv("APP_ENV")
# if app_env:
#     dotenv_path = f".env.{app_env}"
#     print(dotenv_path)
#     load_dotenv(
#         dotenv_path=dotenv_path, override=True
#     )  # override=True 特定环境文件覆盖 .env
load_dotenv(dotenv_path=".env.dev", override=True)
logger.info("APP_ENV: .env.dev")

# LANGCHAIN
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = (
    f"test-{datetime.now().strftime('%Y.%m.%d:%H')}"  # 自定义用例名称,使用当前日期:XX时
)

# --- 导入数据库管理器 ---
from src.config.database_manager import (
    close_databases,
    get_database_manager,
    init_databases,
)
from src.models.user import User  # 导入 User 模型
from src.service.knowledgeSev import load_all_knowledge_bases_to_cache
from src.utils.pwdHash import get_password_hash  # 导入密码哈希函数


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期管理 - 使用新的数据库管理器（开发版本）"""

    # === 应用启动阶段 ===
    logger.info("应用程序启动：开始初始化各项服务...")

    try:
        # 1. 初始化数据库连接（使用新的管理器）
        logger.info("正在初始化数据库连接管理器...")
        await init_databases()
        logger.info("数据库连接管理器初始化完成")

        # 2. 预加载知识库缓存
        try:
            manager = await get_database_manager()
            await manager.get_redis_client()  # 确认 Redis 客户端可用
            logger.info("Redis 客户端获取成功，正在预加载知识库缓存...")
            await load_all_knowledge_bases_to_cache()
            logger.info("知识库缓存预加载完成")
        except RuntimeError as e:
            logger.warning(f"无法获取 Redis 客户端，跳过知识库缓存预加载: {e}")
        except Exception as e:
            logger.error(f"预加载知识库缓存时发生错误: {e}", exc_info=True)

        # 3. 创建 root 用户（如果不存在）
        try:
            root_user = await User.find_one(User.username == "root")
            if not root_user:
                hashed_password = get_password_hash("123456")
                root_user = User(
                    username="root",
                    password=hashed_password,
                    email="root@example.com",
                )
                await root_user.create()
                logger.info("Root 用户创建成功")
            else:
                logger.info("Root 用户已存在")
        except Exception as e:
            logger.error(f"创建 Root 用户时出错: {e}")

        # 4. 启动 MCP Client
        logger.info("正在初始化 MCP 客户端...")
        await startup_mcp_client()  # 这个 await 会在 ProactorEventLoop 下执行
        logger.info("MCP 客户端初始化完成")

        # 5. 执行健康检查
        try:
            manager = await get_database_manager()
            health_status = await manager.health_check(force=True)
            logger.info(f"启动时健康检查完成: {health_status}")
        except Exception as e:
            logger.warning(f"启动时健康检查失败: {e}")

        logger.info("应用程序启动完成，所有服务已就绪")

    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        raise

    # === 应用运行阶段 ===
    yield

    # === 应用关闭阶段 ===
    logger.info("应用程序关闭：开始清理各项服务...")

    try:
        # 1. 关闭 MCP Client
        logger.info("正在关闭 MCP 客户端...")
        await shutdown_mcp_client()
        logger.info("MCP 客户端已关闭")

        # 2. 关闭数据库连接（使用新的管理器）
        logger.info("正在关闭数据库连接...")
        await close_databases()
        logger.info("数据库连接已关闭")

        logger.info("应用程序已成功关闭")

    except Exception as e:
        logger.error(f"应用程序关闭时发生错误: {e}", exc_info=True)


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
from src.router.healthRouter import HealthRouter
from src.router.knowledgeRouter import knowledgeRouter
from src.router.sessionRouter import SessionRouter
from src.router.userRouter import UserRouter
from src.utils.rag_tools import testRouter

app.include_router(router=AuthRouter)
app.include_router(router=UserRouter, prefix="/user", tags=["user"])
app.include_router(router=ChatRouter, prefix="/chat", tags=["chat"])
app.include_router(router=knowledgeRouter, prefix="/knowledge", tags=["knowledge"])
app.include_router(router=SessionRouter, prefix="/session", tags=["session"])
app.include_router(router=AssistantRouter, prefix="/assistant", tags=["assistant"])
app.include_router(router=AgentRouter, prefix="/agent", tags=["agent"])
app.include_router(router=HealthRouter, prefix="/db", tags=["database", "health"])
app.include_router(router=testRouter, prefix="/test", tags=["test"])


# 当访问路径为/ ，重定向路由到/docs
@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


#  静态文件  -----------------------------------------------------------
# from fastapi.staticfiles import StaticFiles

# app.mount("/static", StaticFiles(directory="static"), name="static")


# 将本项目设为MCP服务器
# 显式的 operation_id（工具将被命名为 "current_datetime"）
@app.get("/current_datetime", operation_id="current_datetime")
def currentDatetime():
    # 获取当前时间
    now = datetime.now()

    # 格式化为字符串（默认格式）
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"message": formatted_time}


from fastapi_mcp import FastApiMCP

mcp_app = FastApiMCP(
    app,
    # Optional parameters
    name="RAG_MCP",
    description="RAG知识库",
)
# 将 MCP 服务器挂载到 FastAPI 应用
mcp_app.mount()


# 启动web服务 ----------------------------------------------------------
import uvicorn


# --- 自定义服务器类 解决Windows上 FastAPI/Asyncio 子进程 `NotImplementedError`
class ProactorServer(uvicorn.Server):
    def run(self, sockets=None):
        # 在服务器运行前设置事件循环策略 (仅 Windows)
        # 如果在脚本顶部已经设置了全局策略, 这里的再次设置是可选的，
        # asyncio.run() 将会使用在它被调用之前最后设置的有效策略来创建新的事件循环。
        if sys.platform == "win32":
            # Potentially redundant if set globally at the top, but ensures policy for this specific run context.
            # print("Re-affirming ProactorEventLoopPolicy for Uvicorn server on Windows within ProactorServer.run().")
            # asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # Can be commented out if global set is sufficient
            pass  # Global policy should be picked up by asyncio.run()
        # 使用 asyncio.run 启动服务器的 serve 方法
        asyncio.run(self.serve(sockets=sockets))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # 默认端口设置为8080
    host = os.getenv("HOST", "127.0.0.1")  # 默认主机设置为127.0.0.1
    # uvicorn.run("main_dev:app", host=host, port=port)
    # uvicorn.run("main_dev:app", host=host, port=port, reload=True)

    # 如果启用 MCP
    # --- 修改服务器启动方式 ---
    print(f"Starting MCP Agent server with ProactorServer at http://{host}:{port}")

    # 1. 创建 Uvicorn 配置，确保 reload=False
    config = uvicorn.Config(app="main_dev:app", host=host, port=port, reload=False)

    # 2. 实例化自定义服务器
    server = ProactorServer(config=config)
    server.run()
