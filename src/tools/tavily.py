"""Tavily search tool wrapper.

This module provides a centralized TavilySearch import with conditional
availability checking and a factory function for creating Tavily tools.
"""

import logging

# 使用条件导入来避免 aiohttp 循环导入问题
try:
    from langchain_tavily import TavilySearch

    TAVILY_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入 TavilySearch，将使用备用搜索工具: {e}")
    TavilySearch = None
    TAVILY_AVAILABLE = False


def create_tavily_tool(max_results: int = 2):
    """Create a TavilySearch tool instance.

    Args:
        max_results: Maximum number of search results to return (default: 2)

    Returns:
        TavilySearch instance if available, None otherwise
    """
    if TAVILY_AVAILABLE and TavilySearch is not None:
        try:
            tool = TavilySearch(max_results=max_results)
            logging.info(f"成功创建 TavilySearch 工具 (max_results={max_results})")
            return tool
        except Exception as e:
            logging.warning(f"创建 TavilySearch 失败: {e}")
            return None
    else:
        logging.info("TavilySearch 不可用")
        return None


__all__ = [
    "TavilySearch",
    "TAVILY_AVAILABLE",
    "create_tavily_tool",
]
