import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

# 配置日志记录器
logger = logging.getLogger(__name__)

# SiliconFlow API 的基础 URL
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/rerank"


async def call_siliconflow_rerank(
    api_key: str,
    query: str,
    documents: List[str],
    model: str = "BAAI/bge-reranker-v2-m3",
    top_n: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    异步调用 SiliconFlow 的 Rerank API。

    Args:
        api_key (str): SiliconFlow 的 API 密钥。
        query (str): 用户的查询语句。
        documents (List[str]): 需要重排序的文档内容列表。
        model (str): 使用的 Rerank 模型名称。默认为 "BAAI/bge-reranker-v2-m3"。
                     根据 SiliconFlow 文档，可选: "BAAI/bge-reranker-v2-m3",
                     "Pro/BAAI/bge-reranker-v2-m3", "netease-youdao/bce-reranker-base_v1"
        top_n (Optional[int]): 需要返回的最相关文档数量。如果为 None，API 会使用其默认值。

    Returns:
        Optional[List[Dict[str, Any]]]: 排序后的结果列表，包含 'index' 和 'relevance_score'。
                                         如果 API 调用失败或返回非预期格式，则返回 None。
                                         每个字典形如: {'index': int, 'relevance_score': float}
                                         其中 'index' 是原始 documents 列表中的索引。
    """
    if not api_key:
        logger.error("SiliconFlow API key 未提供，无法调用 Rerank 服务。")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,  # 通常我们只需要排序后的索引和分数
    }
    if top_n is not None:
        payload["top_n"] = top_n

    logger.debug(
        f"向 SiliconFlow Rerank API 发送请求: URL={SILICONFLOW_API_URL}, Model={model}, Query='{query[:50]}...', Docs Count={len(documents)}"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SILICONFLOW_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0,  # 设置超时时间 (秒)
            )

            response.raise_for_status()  # 如果状态码不是 2xx，则抛出 HTTPStatusError

            result = response.json()
            logger.debug(f"收到 SiliconFlow Rerank API 响应: {result}")

            # 验证响应结构并提取所需信息
            if "results" in result and isinstance(result["results"], list):
                ranked_results = []
                for item in result["results"]:
                    index = item.get("index")
                    score = item.get("relevance_score")
                    if index is not None and score is not None:
                        ranked_results.append(
                            {"index": index, "relevance_score": score}
                        )
                    else:
                        logger.warning(
                            f"SiliconFlow 响应中的项目缺少 index 或 relevance_score: {item}"
                        )

                # 根据 relevance_score 降序排序 (API 可能已经排序，但最好确认)
                ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                return ranked_results
            else:
                logger.error(f"SiliconFlow Rerank API 响应格式不符合预期: {result}")
                return None

    except httpx.HTTPStatusError as e:
        logger.error(
            f"调用 SiliconFlow Rerank API 时发生 HTTP 错误: {e.response.status_code} - {e.response.text}"
        )
        return None
    except httpx.RequestError as e:
        logger.error(f"调用 SiliconFlow Rerank API 时发生请求错误: {e}")
        return None
    except Exception as e:
        logger.error(f"调用 SiliconFlow Rerank API 时发生未知错误: {e}", exc_info=True)
        return None


