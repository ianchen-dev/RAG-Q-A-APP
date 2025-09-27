"""
OneAPI健康检查工具模块
提供OneAPI服务连通性检查和模型列表获取功能
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class OneAPIHealthChecker:
    """OneAPI健康检查器"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        初始化OneAPI健康检查器

        Args:
            base_url: OneAPI基础URL，如果不提供则从环境变量获取
            api_key: API密钥，如果不提供则从环境变量获取
        """
        self.base_url = base_url or os.getenv("ONEAPI_BASE_URL")
        self.api_key = api_key or os.getenv("ONEAPI_API_KEY")

        # 清理base_url，确保格式正确
        if self.base_url and self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")

    async def check_connection(self, timeout: int = 10) -> Dict[str, Any]:
        """
        检查OneAPI连接状态

        Args:
            timeout: 超时时间（秒）

        Returns:
            检查结果字典
        """
        result = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": self.base_url,
            "error": None,
            "response_time_ms": None,
            "models_count": 0,
            "available_models": [],
        }

        # 检查配置
        if not self.base_url:
            result.update(
                {"status": "misconfigured", "error": "ONEAPI_BASE_URL not configured"}
            )
            return result

        if not self.api_key:
            result.update(
                {"status": "misconfigured", "error": "ONEAPI_API_KEY not configured"}
            )
            return result

        try:
            start_time = datetime.now()

            # 测试模型列表接口
            models_url = f"{self.base_url}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(models_url, headers=headers) as response:
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000
                    result["response_time_ms"] = round(response_time, 2)

                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", [])

                        result.update(
                            {
                                "status": "healthy",
                                "models_count": len(models),
                                "available_models": [
                                    model.get("id", "unknown") for model in models[:10]
                                ],  # 限制返回前10个模型
                            }
                        )

                        logger.info(
                            f"OneAPI健康检查成功: 响应时间 {response_time:.2f}ms, 可用模型数量: {len(models)}"
                        )

                    else:
                        error_text = await response.text()
                        result.update(
                            {
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}: {error_text}",
                            }
                        )
                        logger.warning(f"OneAPI健康检查失败: HTTP {response.status}")

        except asyncio.TimeoutError:
            result.update(
                {
                    "status": "timeout",
                    "error": f"Connection timeout after {timeout} seconds",
                }
            )
            logger.error(f"OneAPI连接超时: {timeout}秒")

        except aiohttp.ClientConnectorError as e:
            result.update(
                {"status": "connection_failed", "error": f"Connection failed: {str(e)}"}
            )
            logger.error(f"OneAPI连接失败: {e}")

        except Exception as e:
            result.update({"status": "error", "error": f"Unexpected error: {str(e)}"})
            logger.error(f"OneAPI健康检查异常: {e}", exc_info=True)

        return result

    async def check_embeddings_model(
        self, model_name: str = "text-embedding-ada-002", timeout: int = 15
    ) -> Dict[str, Any]:
        """
        检查嵌入模型的可用性

        Args:
            model_name: 要测试的嵌入模型名称
            timeout: 超时时间（秒）

        Returns:
            检查结果字典
        """
        result = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "error": None,
            "response_time_ms": None,
        }

        if not self.base_url or not self.api_key:
            result.update(
                {
                    "status": "misconfigured",
                    "error": "OneAPI base_url or api_key not configured",
                }
            )
            return result

        try:
            start_time = datetime.now()

            # 测试嵌入接口
            embeddings_url = f"{self.base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {"model": model_name, "input": "Test embedding connectivity"}

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.post(
                    embeddings_url, headers=headers, json=data
                ) as response:
                    end_time = datetime.now()
                    response_time = (end_time - start_time).total_seconds() * 1000
                    result["response_time_ms"] = round(response_time, 2)

                    if response.status == 200:
                        response_data = await response.json()
                        embeddings = response_data.get("data", [])

                        if embeddings and embeddings[0].get("embedding"):
                            result["status"] = "healthy"
                            logger.info(
                                f"OneAPI嵌入模型 {model_name} 检查成功: 响应时间 {response_time:.2f}ms"
                            )
                        else:
                            result.update(
                                {
                                    "status": "unhealthy",
                                    "error": "Empty or invalid embedding response",
                                }
                            )
                    else:
                        error_text = await response.text()
                        result.update(
                            {
                                "status": "unhealthy",
                                "error": f"HTTP {response.status}: {error_text}",
                            }
                        )

        except asyncio.TimeoutError:
            result.update(
                {
                    "status": "timeout",
                    "error": f"Embedding test timeout after {timeout} seconds",
                }
            )

        except Exception as e:
            result.update(
                {"status": "error", "error": f"Embedding test error: {str(e)}"}
            )
            logger.error(f"OneAPI嵌入模型检查异常: {e}", exc_info=True)

        return result


# 全局实例
_oneapi_checker = None


def get_oneapi_checker() -> OneAPIHealthChecker:
    """获取OneAPI健康检查器实例"""
    global _oneapi_checker
    if _oneapi_checker is None:
        _oneapi_checker = OneAPIHealthChecker()
    return _oneapi_checker


async def check_oneapi_health(
    include_embeddings: bool = True, timeout: int = 10
) -> Dict[str, Any]:
    """
    执行完整的OneAPI健康检查

    Args:
        include_embeddings: 是否包含嵌入模型检查
        timeout: 超时时间（秒）

    Returns:
        完整的健康检查结果
    """
    checker = get_oneapi_checker()

    # 基础连接检查
    connection_result = await checker.check_connection(timeout=timeout)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "connection": connection_result,
        "embeddings": None,
    }

    # 如果基础连接正常且需要检查嵌入模型
    if include_embeddings and connection_result["status"] == "healthy":
        embeddings_result = await checker.check_embeddings_model(timeout=timeout + 5)
        result["embeddings"] = embeddings_result

    # 确定总体状态
    overall_status = connection_result["status"]
    if include_embeddings and result["embeddings"]:
        if overall_status == "healthy" and result["embeddings"]["status"] != "healthy":
            overall_status = "partial"  # 连接正常但嵌入模型有问题

    result["overall_status"] = overall_status

    return result


# 测试函数
async def _test_oneapi_health():
    """测试OneAPI健康检查功能"""
    print("=== OneAPI健康检查测试 ===")

    result = await check_oneapi_health(include_embeddings=True)

    print(f"总体状态: {result['overall_status']}")
    print(f"连接检查: {result['connection']['status']}")

    if result["connection"]["status"] == "healthy":
        print(f"响应时间: {result['connection']['response_time_ms']}ms")
        print(f"可用模型数量: {result['connection']['models_count']}")
        print(f"部分模型列表: {result['connection']['available_models']}")
    else:
        print(f"连接错误: {result['connection'].get('error', 'Unknown error')}")

    if result["embeddings"]:
        print(f"嵌入模型检查: {result['embeddings']['status']}")
        if result["embeddings"]["status"] == "healthy":
            print(f"嵌入响应时间: {result['embeddings']['response_time_ms']}ms")
        else:
            print(f"嵌入错误: {result['embeddings'].get('error', 'Unknown error')}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(_test_oneapi_health())


