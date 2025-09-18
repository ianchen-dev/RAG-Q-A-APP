"""
Chroma向量数据库适配器实现
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.adapters.vector_db_adapter import VectorDBAdapter
from src.config.vector_db_config import get_vector_db_config

logger = logging.getLogger(__name__)


class ChromaAdapter(VectorDBAdapter):
    """Chroma向量数据库适配器"""

    def __init__(self, embeddings: Embeddings):
        """
        初始化Chroma适配器

        Args:
            embeddings: 嵌入模型实例
        """
        super().__init__(embeddings)
        self.config = get_vector_db_config().chroma
        self._collections: Dict[str, Chroma] = {}  # 缓存已加载的集合
        logger.info(
            f"ChromaAdapter 初始化完成，持久化目录: {self.config.persist_directory}"
        )

    def _get_collection_path(self, collection_name: str) -> str:
        """获取集合的持久化路径"""
        return os.path.join(self.config.persist_directory, collection_name)

    def _get_or_create_collection(self, collection_name: str) -> Chroma:
        """获取或创建Chroma集合实例"""
        if collection_name in self._collections:
            return self._collections[collection_name]

        persist_directory = self._get_collection_path(collection_name)

        chroma_instance = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_metadata=self.config.collection_metadata,
        )

        self._collections[collection_name] = chroma_instance
        logger.debug(f"Chroma集合 '{collection_name}' 实例已创建/加载")
        return chroma_instance

    async def create_collection(self, collection_name: str) -> bool:
        """
        创建Chroma集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 创建是否成功
        """
        try:
            # Chroma在第一次访问时自动创建集合
            self._get_or_create_collection(collection_name)
            logger.info(f"Chroma集合 '{collection_name}' 创建成功")
            return True
        except Exception as e:
            logger.error(f"创建Chroma集合 '{collection_name}' 失败: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查Chroma集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 集合是否存在
        """
        persist_directory = self._get_collection_path(collection_name)
        return os.path.isdir(persist_directory)

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """
        向Chroma集合添加文档

        Args:
            collection_name: 集合名称
            documents: 文档列表

        Returns:
            List[str]: 文档ID列表
        """
        if not documents:
            logger.warning(f"尝试向集合 '{collection_name}' 添加空文档列表")
            return []

        try:
            collection = self._get_or_create_collection(collection_name)

            # 检查集合是否已存在
            if await self.collection_exists(collection_name):
                # 集合已存在，使用add_documents
                document_ids = await collection.aadd_documents(documents)
            else:
                # 集合不存在，使用from_documents创建
                collection = await Chroma.afrom_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=self._get_collection_path(collection_name),
                    collection_metadata=self.config.collection_metadata,
                )
                self._collections[collection_name] = collection
                # Chroma.afrom_documents不返回IDs，我们需要生成
                document_ids = [f"doc_{i}" for i in range(len(documents))]

            logger.info(
                f"成功向Chroma集合 '{collection_name}' 添加 {len(documents)} 个文档"
            )
            return document_ids
        except Exception as e:
            logger.error(f"向Chroma集合 '{collection_name}' 添加文档失败: {e}")
            raise

    async def delete_documents(
        self,
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        从Chroma集合删除文档

        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件
            document_ids: 文档ID列表

        Returns:
            bool: 删除是否成功
        """
        try:
            if not await self.collection_exists(collection_name):
                logger.warning(f"尝试从不存在的集合 '{collection_name}' 删除文档")
                return False

            collection = self._get_or_create_collection(collection_name)

            if document_ids:
                # 根据ID删除
                collection.delete(ids=document_ids)
            elif filter_dict:
                # 根据过滤条件删除
                collection.delete(where=filter_dict)
            else:
                logger.warning("删除文档时未提供document_ids或filter_dict")
                return False

            logger.info(f"成功从Chroma集合 '{collection_name}' 删除文档")
            return True
        except Exception as e:
            logger.error(f"从Chroma集合 '{collection_name}' 删除文档失败: {e}")
            return False

    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        在Chroma集合中搜索文档

        Args:
            collection_name: 集合名称
            query: 查询文本
            limit: 返回结果数量限制
            filter_dict: 过滤条件

        Returns:
            List[Document]: 搜索结果文档列表
        """
        try:
            if not await self.collection_exists(collection_name):
                logger.warning(f"尝试在不存在的集合 '{collection_name}' 中搜索")
                return []

            collection = self._get_or_create_collection(collection_name)

            # 构建搜索参数
            search_kwargs = {"k": limit}
            if filter_dict:
                search_kwargs["filter"] = filter_dict

            # 创建检索器并搜索
            retriever = collection.as_retriever(search_kwargs=search_kwargs)
            documents = await retriever.ainvoke(query)

            logger.debug(
                f"在Chroma集合 '{collection_name}' 中搜索到 {len(documents)} 个文档"
            )
            return documents
        except Exception as e:
            logger.error(f"在Chroma集合 '{collection_name}' 中搜索失败: {e}")
            return []

    async def get_all_documents(
        self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        获取Chroma集合中的所有文档

        Args:
            collection_name: 集合名称
            filter_dict: 过滤条件

        Returns:
            List[Document]: 文档列表
        """
        try:
            if not await self.collection_exists(collection_name):
                logger.warning(f"尝试从不存在的集合 '{collection_name}' 获取文档")
                return []

            collection = self._get_or_create_collection(collection_name)

            # 使用get方法获取所有文档
            result = collection.get(
                where=filter_dict, include=["documents", "metadatas"]
            )

            # 转换为Document对象列表
            documents = []
            doc_contents = result.get("documents", [])
            metadatas = result.get("metadatas", [])

            for content, metadata in zip(doc_contents, metadatas):
                documents.append(
                    Document(
                        page_content=content, metadata=metadata if metadata else {}
                    )
                )

            logger.debug(
                f"从Chroma集合 '{collection_name}' 获取到 {len(documents)} 个文档"
            )
            return documents
        except Exception as e:
            logger.error(f"从Chroma集合 '{collection_name}' 获取文档失败: {e}")
            return []

    def get_retriever(
        self,
        collection_name: str,
        search_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        获取Chroma检索器

        Args:
            collection_name: 集合名称
            search_k: 检索结果数量
            filter_dict: 过滤条件

        Returns:
            BaseRetriever: Langchain检索器实例
        """
        collection = self._get_or_create_collection(collection_name)

        search_kwargs = {"k": search_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return collection.as_retriever(search_kwargs=search_kwargs)

    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除Chroma集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 删除是否成功
        """
        try:
            import shutil

            # 从缓存中移除
            if collection_name in self._collections:
                del self._collections[collection_name]

            # 删除持久化目录
            persist_directory = self._get_collection_path(collection_name)
            if os.path.isdir(persist_directory):
                shutil.rmtree(persist_directory)
                logger.info(
                    f"Chroma集合 '{collection_name}' 目录已删除: {persist_directory}"
                )

            logger.info(f"Chroma集合 '{collection_name}' 删除成功")
            return True
        except Exception as e:
            logger.error(f"删除Chroma集合 '{collection_name}' 失败: {e}")
            return False

    async def close(self):
        """关闭连接，清理资源"""
        # Chroma不需要显式关闭连接
        self._collections.clear()
        logger.debug("ChromaAdapter 资源已清理")
