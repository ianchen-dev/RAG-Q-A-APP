"""
Milvus向量数据库适配器实现
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.adapters.vector_db_adapter import VectorDBAdapter
from src.config.vector_db_config import get_vector_db_config

logger = logging.getLogger(__name__)


class MilvusAdapter(VectorDBAdapter):
    """Milvus向量数据库适配器"""

    def __init__(self, embeddings: Embeddings):
        """
        初始化Milvus适配器

        Args:
            embeddings: 嵌入模型实例
        """
        super().__init__(embeddings)
        self.config = get_vector_db_config().milvus
        self._client = None
        self._collections: Dict[str, Any] = {}  # 缓存已加载的集合
        logger.info(
            f"MilvusAdapter 初始化完成，服务器: {self.config.host}:{self.config.port}"
        )

    def _get_client(self):
        """获取Milvus客户端连接"""
        if self._client is None:
            try:
                from pymilvus import connections, utility

                # 建立连接
                connections.connect(
                    alias="default",
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.user if self.config.user else None,
                    password=self.config.password if self.config.password else None,
                    secure=self.config.secure,
                    db_name=self.config.db_name,
                )

                self._client = connections
                logger.info(
                    f"Milvus连接建立成功: {self.config.host}:{self.config.port}"
                )
            except ImportError:
                raise ImportError(
                    "使用Milvus适配器需要安装pymilvus库。请运行: pip install pymilvus"
                )
            except Exception as e:
                logger.error(f"连接Milvus失败: {e}")
                raise

        return self._client

    def _get_collection(self, collection_name: str):
        """获取Milvus集合实例"""
        if collection_name in self._collections:
            return self._collections[collection_name]

        try:
            from pymilvus import Collection

            self._get_client()  # 确保连接已建立
            collection = Collection(collection_name)
            self._collections[collection_name] = collection
            logger.debug(f"Milvus集合 '{collection_name}' 实例已加载")
            return collection
        except Exception as e:
            logger.error(f"获取Milvus集合 '{collection_name}' 失败: {e}")
            raise

    async def create_collection(self, collection_name: str) -> bool:
        """
        创建Milvus集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 创建是否成功
        """
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

            self._get_client()

            # 检查集合是否已存在
            if await self.collection_exists(collection_name):
                logger.info(f"Milvus集合 '{collection_name}' 已存在")
                return True

            # 获取嵌入向量维度
            test_embedding = await self.embeddings.aembed_query("test")
            vector_dim = len(test_embedding)

            # 定义集合schema
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
                ),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            ]

            schema = CollectionSchema(
                fields, description=f"Collection for {collection_name}"
            )

            # 创建集合
            collection = Collection(collection_name, schema)

            # 创建索引
            index_params = {
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                "params": {"nlist": self.config.nlist},
            }
            collection.create_index("vector", index_params)

            # 加载集合到内存
            collection.load()

            self._collections[collection_name] = collection
            logger.info(
                f"Milvus集合 '{collection_name}' 创建成功，向量维度: {vector_dim}"
            )
            return True

        except Exception as e:
            logger.error(f"创建Milvus集合 '{collection_name}' 失败: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查Milvus集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 集合是否存在
        """
        try:
            from pymilvus import utility

            self._get_client()
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"检查Milvus集合 '{collection_name}' 是否存在失败: {e}")
            return False

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """
        向Milvus集合添加文档

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
            # 确保集合存在
            if not await self.collection_exists(collection_name):
                await self.create_collection(collection_name)

            collection = self._get_collection(collection_name)

            # 生成文档ID
            doc_ids = [str(uuid.uuid4()) for _ in documents]

            # 获取嵌入向量
            contents = [doc.page_content for doc in documents]
            vectors = await self.embeddings.aembed_documents(contents)

            # 准备插入数据
            entities = [doc_ids, contents, [doc.metadata for doc in documents], vectors]

            # 插入数据
            collection.insert(entities)
            collection.flush()  # 确保数据持久化

            logger.info(
                f"成功向Milvus集合 '{collection_name}' 添加 {len(documents)} 个文档"
            )
            return doc_ids

        except Exception as e:
            logger.error(f"向Milvus集合 '{collection_name}' 添加文档失败: {e}")
            raise

    async def delete_documents(
        self,
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        从Milvus集合删除文档

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

            collection = self._get_collection(collection_name)

            if document_ids:
                # 根据ID删除
                expr = f"id in {document_ids}"
                collection.delete(expr)
            elif filter_dict:
                # 根据过滤条件删除（需要构建表达式）
                expr = self._build_filter_expr(filter_dict)
                collection.delete(expr)
            else:
                logger.warning("删除文档时未提供document_ids或filter_dict")
                return False

            collection.flush()
            logger.info(f"成功从Milvus集合 '{collection_name}' 删除文档")
            return True

        except Exception as e:
            logger.error(f"从Milvus集合 '{collection_name}' 删除文档失败: {e}")
            return False

    def _build_filter_expr(self, filter_dict: Dict[str, Any]) -> str:
        """构建Milvus过滤表达式"""
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append(f'metadata["{key}"] == "{value}"')
            elif isinstance(value, (int, float)):
                conditions.append(f'metadata["{key}"] == {value}')
            elif isinstance(value, list):
                value_str = ", ".join(
                    [f'"{v}"' if isinstance(v, str) else str(v) for v in value]
                )
                conditions.append(f'metadata["{key}"] in [{value_str}]')

        return " and ".join(conditions) if conditions else "id != ''"

    async def search_documents(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        在Milvus集合中搜索文档

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

            collection = self._get_collection(collection_name)

            # 获取查询向量
            query_vector = await self.embeddings.aembed_query(query)

            # 构建搜索参数
            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"nprobe": min(self.config.nlist, 16)},
            }

            # 构建过滤表达式
            expr = None
            if filter_dict:
                expr = self._build_filter_expr(filter_dict)

            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["content", "metadata"],
            )

            # 转换结果为Document对象
            documents = []
            for hits in results:
                for hit in hits:
                    documents.append(
                        Document(
                            page_content=hit.entity.get("content"),
                            metadata=hit.entity.get("metadata", {}),
                        )
                    )

            logger.debug(
                f"在Milvus集合 '{collection_name}' 中搜索到 {len(documents)} 个文档"
            )
            return documents

        except Exception as e:
            logger.error(f"在Milvus集合 '{collection_name}' 中搜索失败: {e}")
            return []

    async def get_all_documents(
        self, collection_name: str, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        获取Milvus集合中的所有文档

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

            collection = self._get_collection(collection_name)

            # 构建查询表达式
            expr = None
            if filter_dict:
                expr = self._build_filter_expr(filter_dict)

            # 查询所有文档
            results = collection.query(
                expr=expr or "id != ''", output_fields=["content", "metadata"]
            )

            # 转换为Document对象列表
            documents = []
            for result in results:
                documents.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                    )
                )

            logger.debug(
                f"从Milvus集合 '{collection_name}' 获取到 {len(documents)} 个文档"
            )
            return documents

        except Exception as e:
            logger.error(f"从Milvus集合 '{collection_name}' 获取文档失败: {e}")
            return []

    def get_retriever(
        self,
        collection_name: str,
        search_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> BaseRetriever:
        """
        获取Milvus检索器

        Args:
            collection_name: 集合名称
            search_k: 检索结果数量
            filter_dict: 过滤条件

        Returns:
            BaseRetriever: Langchain检索器实例
        """
        # 创建自定义检索器，封装Milvus搜索逻辑
        return MilvusRetriever(
            adapter=self,
            collection_name=collection_name,
            search_k=search_k,
            filter_dict=filter_dict,
        )

    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除Milvus集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 删除是否成功
        """
        try:
            from pymilvus import utility

            self._get_client()

            # 从缓存中移除
            if collection_name in self._collections:
                del self._collections[collection_name]

            # 删除集合
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Milvus集合 '{collection_name}' 删除成功")
            else:
                logger.info(f"Milvus集合 '{collection_name}' 不存在，无需删除")

            return True

        except Exception as e:
            logger.error(f"删除Milvus集合 '{collection_name}' 失败: {e}")
            return False

    async def close(self):
        """关闭连接，清理资源"""
        try:
            if self._client:
                from pymilvus import connections

                connections.disconnect("default")
                self._client = None

            self._collections.clear()
            logger.debug("MilvusAdapter 连接已关闭，资源已清理")
        except Exception as e:
            logger.error(f"关闭MilvusAdapter连接时出错: {e}")


class MilvusRetriever(BaseRetriever):
    """Milvus自定义检索器"""

    def __init__(
        self,
        adapter: MilvusAdapter,
        collection_name: str,
        search_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化Milvus检索器

        Args:
            adapter: Milvus适配器实例
            collection_name: 集合名称
            search_k: 检索结果数量
            filter_dict: 过滤条件
        """
        super().__init__()
        self.adapter = adapter
        self.collection_name = collection_name
        self.search_k = search_k
        self.filter_dict = filter_dict

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """同步获取相关文档"""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.adapter.search_documents(
                self.collection_name, query, self.search_k, self.filter_dict
            )
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步获取相关文档"""
        return await self.adapter.search_documents(
            self.collection_name, query, self.search_k, self.filter_dict
        )
