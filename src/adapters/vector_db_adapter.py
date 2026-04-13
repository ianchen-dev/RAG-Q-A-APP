"""
向量数据库适配器接口
定义了统一的向量数据库操作接口，支持不同的向量数据库实现
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


# 向量数据库适配器抽象基类
class VectorDBAdapter(ABC):
    """向量数据库适配器抽象基类"""

    def __init__(self, embeddings: Embeddings):
        """
        初始化适配器

        Args:
            embeddings: 嵌入模型实例
        """
        self.embeddings = embeddings

    @abstractmethod
    async def create_collection(self, collection_name: str) -> bool:
        """
        创建集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 创建是否成功
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 集合是否存在
        """
        pass

    @abstractmethod
    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """
        向集合添加文档

        Args:
            collection_name: 集合名称
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        pass

    @abstractmethod
    async def delete_documents(
        self,
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        删除文档

        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件
            document_ids: 文档ID列表

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        搜索文档

        Args:
            collection_name: 集合名称
            query: 查询文本
            limit: 返回结果数量限制
            filter_dict: 过滤条件

        Returns:
            List[Document]: 搜索结果文档列表
        """
        pass

    @abstractmethod
    async def get_all_documents(
        self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        获取集合中的所有文档

        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件

        Returns:
            List[Document]: 文档列表
        """
        pass

    @abstractmethod
    def get_retriever(
        self,
        collection_name: str,
        search_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        获取检索器

        Args:
            collection_name: 集合名称
            search_k: 检索结果数量
            filter_dict: 过滤条件

        Returns:
            BaseRetriever: Langchain检索器实例
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 删除是否成功
        """
        pass

    @abstractmethod
    async def close(self):
        """关闭连接，清理资源"""
        pass

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()


# 向量存储适配器，用于封装不同向量数据库的VectorStore
class VectorStoreAdapter:
    """向量存储适配器，用于封装不同向量数据库的VectorStore"""

    def __init__(self, adapter: VectorDBAdapter, collection_name: str):
        """
        初始化向量存储适配器

        Args:
            adapter: 向量数据库适配器
            collection_name: 集合名称
        """
        self.adapter = adapter
        self.collection_name = collection_name

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档"""
        return await self.adapter.add_documents(self.collection_name, documents)

    async def aadd_documents(self, documents: List[Document]) -> List[str]:
        """异步添加文档（兼容Langchain接口）"""
        return await self.add_documents(documents)

    def as_retriever(
        self, search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """转换为检索器"""
        search_kwargs = search_kwargs or {}
        search_k = search_kwargs.get("k", 3)
        filter_dict = search_kwargs.get("filter", None)
        return self.adapter.get_retriever(self.collection_name, search_k, filter_dict)

    async def delete(self, filter_dict: Optional[Dict[str, Any]] = None):
        """删除文档"""
        await self.adapter.delete_documents(self.collection_name, filter_dict)

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        获取文档（同步方法，用于兼容现有代码）
        注意：这是一个同步方法，内部需要处理异步调用
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if asyncio.iscoroutinefunction(self.adapter.get_all_documents):
            documents = loop.run_until_complete(
                self.adapter.get_all_documents(self.collection_name, where)
            )
        else:
            documents = self.adapter.get_all_documents(self.collection_name, where)

        # 转换为Chroma格式的返回值
        doc_contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        result = {
            "documents": doc_contents,
            "metadatas": metadatas,
            "ids": [f"doc_{i}" for i in range(len(documents))],
        }

        return result
