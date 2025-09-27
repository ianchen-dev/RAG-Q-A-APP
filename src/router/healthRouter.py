import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.config.database_manager import get_database_manager
from src.utils.oneapi_health import check_oneapi_health, get_oneapi_checker

logger = logging.getLogger(__name__)

# 创建路由器
HealthRouter = APIRouter()


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(..., description="总体状态")
    timestamp: str = Field(..., description="检查时间")
    services: Dict[str, Any] = Field(..., description="各服务状态")


class ConnectionStatsResponse(BaseModel):
    """连接统计响应模型"""

    mongodb: Dict[str, Any] = Field(..., description="MongoDB 连接信息")
    redis: Dict[str, Any] = Field(..., description="Redis 连接信息")


class OneAPIHealthResponse(BaseModel):
    """OneAPI健康检查响应模型"""

    overall_status: str = Field(..., description="总体状态")
    timestamp: str = Field(..., description="检查时间")
    connection: Dict[str, Any] = Field(..., description="连接检查结果")
    embeddings: Optional[Dict[str, Any]] = Field(None, description="嵌入模型检查结果")


@HealthRouter.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="数据库健康检查",
    description="检查 MongoDB 和 Redis 连接状态",
)
async def health_check(
    force: bool = Query(False, description="是否强制执行检查，忽略时间间隔限制"),
):
    """
    执行数据库健康检查

    Args:
        force: 是否强制执行检查

    Returns:
        健康检查结果
    """
    try:
        logger.info(f"开始执行健康检查, force={force}")

        # 获取数据库管理器
        manager = await get_database_manager()

        # 执行健康检查
        health_status = await manager.health_check(force=force)

        # 处理跳过的情况
        if health_status.get("skipped"):
            return HealthCheckResponse(
                status="skipped",
                timestamp=health_status.get("timestamp", ""),
                services={"reason": health_status.get("reason", "未知原因")},
            )

        # 判断总体状态
        mongodb_status = health_status.get("mongodb", {}).get("status", "unknown")
        redis_status = health_status.get("redis", {}).get("status", "unknown")

        if mongodb_status == "healthy" and redis_status == "healthy":
            overall_status = "healthy"
        elif mongodb_status == "healthy" and redis_status in [
            "unhealthy",
            "not_initialized",
        ]:
            overall_status = "partial"  # MongoDB 正常，Redis 异常
        elif mongodb_status == "unhealthy":
            overall_status = "unhealthy"  # MongoDB 异常
        else:
            overall_status = "unknown"

        logger.info(f"健康检查完成，总体状态: {overall_status}")

        return HealthCheckResponse(
            status=overall_status,
            timestamp=health_status.get("timestamp", ""),
            services={
                "mongodb": health_status.get("mongodb", {}),
                "redis": health_status.get("redis", {}),
            },
        )

    except Exception as e:
        logger.error(f"健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@HealthRouter.get(
    "/health/quick",
    summary="快速健康检查",
    description="快速检查数据库管理器状态，不执行实际连接测试",
)
async def quick_health_check():
    """
    快速健康检查，仅检查管理器初始化状态
    """
    try:
        manager = await get_database_manager()

        result = {
            "status": "initialized" if manager._initialized else "not_initialized",
            "mongodb_initialized": manager._mongodb_initialized,
            "redis_initialized": manager._redis_initialized,
        }

        return result

    except Exception as e:
        logger.error(f"快速健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"快速健康检查失败: {str(e)}")


@HealthRouter.get(
    "/stats",
    response_model=ConnectionStatsResponse,
    summary="连接统计信息",
    description="获取数据库连接池统计信息",
)
async def get_connection_stats():
    """
    获取数据库连接统计信息
    """
    try:
        logger.info("获取连接统计信息")

        # 获取数据库管理器
        manager = await get_database_manager()

        # 获取统计信息
        stats = await manager.get_connection_stats()

        logger.info("连接统计信息获取完成")

        return ConnectionStatsResponse(
            mongodb=stats.get("mongodb", {}), redis=stats.get("redis", {})
        )

    except Exception as e:
        logger.error(f"获取连接统计信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取连接统计信息失败: {str(e)}")


@HealthRouter.post(
    "/reconnect", summary="重新连接数据库", description="强制重新初始化数据库连接"
)
async def reconnect_databases():
    """
    重新连接数据库
    用于在数据库连接出现问题时手动触发重连
    """
    try:
        logger.info("开始重新连接数据库")

        # 获取数据库管理器
        manager = await get_database_manager()

        # 先关闭现有连接
        await manager.close()

        # 重新初始化
        await manager.initialize()

        # 执行健康检查确认连接状态
        health_status = await manager.health_check(force=True)

        logger.info("数据库重连完成")

        return {"message": "数据库重连完成", "health_status": health_status}

    except Exception as e:
        logger.error(f"数据库重连失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"数据库重连失败: {str(e)}")


@HealthRouter.get(
    "/config", summary="数据库配置信息", description="获取数据库连接配置信息（已脱敏）"
)
async def get_database_config():
    """
    获取数据库配置信息（脱敏后）
    """
    try:
        logger.info("获取数据库配置信息")

        # 获取数据库管理器
        manager = await get_database_manager()

        # 获取配置信息
        config = {
            "mongodb": {
                "initialized": manager._mongodb_initialized,
                "config": manager._mongodb_config.copy()
                if manager._mongodb_config
                else None,
            },
            "redis": {
                "initialized": manager._redis_initialized,
                "config": manager._redis_config.copy()
                if manager._redis_config
                else None,
            },
        }

        # 脱敏处理
        if config["mongodb"]["config"]:
            config["mongodb"]["config"].pop("url", None)

        if config["redis"]["config"]:
            config["redis"]["config"].pop("password", None)

        logger.info("数据库配置信息获取完成")

        return config

    except Exception as e:
        logger.error(f"获取数据库配置信息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取数据库配置信息失败: {str(e)}")


@HealthRouter.get(
    "/oneapi",
    response_model=OneAPIHealthResponse,
    summary="OneAPI健康检查",
    description="检查OneAPI服务连接状态和模型可用性",
)
async def oneapi_health_check(
    include_embeddings: bool = Query(True, description="是否检查嵌入模型"),
    timeout: int = Query(10, ge=5, le=30, description="超时时间（秒）"),
):
    """
    OneAPI服务健康检查

    Args:
        include_embeddings: 是否包含嵌入模型检查
        timeout: 超时时间（5-30秒）

    Returns:
        OneAPI健康检查结果
    """
    try:
        logger.info(
            f"开始OneAPI健康检查, include_embeddings={include_embeddings}, timeout={timeout}"
        )

        # 执行OneAPI健康检查
        health_result = await check_oneapi_health(
            include_embeddings=include_embeddings, timeout=timeout
        )

        logger.info(f"OneAPI健康检查完成，总体状态: {health_result['overall_status']}")

        return OneAPIHealthResponse(
            overall_status=health_result["overall_status"],
            timestamp=health_result["timestamp"],
            connection=health_result["connection"],
            embeddings=health_result.get("embeddings"),
        )

    except Exception as e:
        logger.error(f"OneAPI健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OneAPI健康检查失败: {str(e)}")


@HealthRouter.get(
    "/oneapi/quick", summary="OneAPI快速检查", description="快速检查OneAPI配置状态"
)
async def oneapi_quick_check():
    """
    OneAPI快速检查，仅验证配置不执行实际连接
    """
    try:
        checker = get_oneapi_checker()

        result = {
            "base_url_configured": bool(checker.base_url),
            "api_key_configured": bool(checker.api_key),
            "base_url": checker.base_url if checker.base_url else "Not configured",
            "api_key_preview": f"{checker.api_key[:8]}..."
            if checker.api_key and len(checker.api_key) > 8
            else "Not configured",
        }

        if result["base_url_configured"] and result["api_key_configured"]:
            result["status"] = "configured"
        else:
            result["status"] = "misconfigured"

        return result

    except Exception as e:
        logger.error(f"OneAPI快速检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OneAPI快速检查失败: {str(e)}")


@HealthRouter.get(
    "/all", summary="全面健康检查", description="检查所有服务状态：数据库、OneAPI等"
)
async def comprehensive_health_check(
    force_db: bool = Query(False, description="是否强制执行数据库检查"),
    include_oneapi_embeddings: bool = Query(
        True, description="OneAPI检查是否包含嵌入模型"
    ),
    timeout: int = Query(10, ge=5, le=30, description="OneAPI检查超时时间（秒）"),
):
    """
    全面健康检查，包括数据库和OneAPI
    """
    try:
        logger.info("开始全面健康检查")

        # 并发执行数据库和OneAPI检查
        import asyncio

        # 数据库检查
        async def db_check():
            try:
                manager = await get_database_manager()
                return await manager.health_check(force=force_db)
            except Exception as e:
                logger.error(f"数据库检查失败: {e}")
                return {"status": "error", "error": str(e)}

        # OneAPI检查
        async def oneapi_check():
            try:
                return await check_oneapi_health(
                    include_embeddings=include_oneapi_embeddings, timeout=timeout
                )
            except Exception as e:
                logger.error(f"OneAPI检查失败: {e}")
                return {"overall_status": "error", "error": str(e)}

        # 并发执行检查
        db_result, oneapi_result = await asyncio.gather(
            db_check(), oneapi_check(), return_exceptions=True
        )

        # 处理异常结果
        if isinstance(db_result, Exception):
            db_result = {"status": "error", "error": str(db_result)}
        if isinstance(oneapi_result, Exception):
            oneapi_result = {"overall_status": "error", "error": str(oneapi_result)}

        # 确定总体状态
        if isinstance(db_result, dict):
            db_status = db_result.get("mongodb", {}).get("status", "unknown")
            redis_status = db_result.get("redis", {}).get("status", "unknown")
        else:
            db_status = "error"
            redis_status = "error"

        if isinstance(oneapi_result, dict):
            oneapi_status = oneapi_result.get("overall_status", "unknown")
        else:
            oneapi_status = "error"

        # 计算总体健康状态
        if all(
            status == "healthy" for status in [db_status, redis_status, oneapi_status]
        ):
            overall_status = "healthy"
        elif any(
            status == "healthy" for status in [db_status, redis_status, oneapi_status]
        ):
            overall_status = "partial"
        else:
            overall_status = "unhealthy"

        from datetime import datetime, timezone

        result = {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": db_result,
            "oneapi": oneapi_result,
        }

        logger.info(f"全面健康检查完成，总体状态: {overall_status}")

        return result

    except Exception as e:
        logger.error(f"全面健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"全面健康检查失败: {str(e)}")
