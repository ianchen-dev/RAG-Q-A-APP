"""
向量数据库工厂类
使用工厂模式创建不同类型的向量数据库适配器
"""

import logging
from typing import Optional

from langchain_core.embeddings import Embeddings

from src.adapters.chroma_adapter import ChromaAdapter
from src.adapters.milvus_adapter import MilvusAdapter
from src.adapters.vector_db_adapter import VectorDBAdapter, VectorStoreAdapter
from src.config.vector_db_config import VectorDBType, get_vector_db_config

logger = logging.getLogger(__name__)


class VectorDBFactory:
    """向量数据库工厂类"""

    @staticmethod
    def create_adapter(
        embeddings: Embeddings,
        db_type: Optional[VectorDBType] = None,
        batch_size: int = 64,
        max_concurrent_batches: int = 5,
    ) -> VectorDBAdapter:
        """
        创建向量数据库适配器

        Args:
            embeddings: 嵌入模型实例
            db_type: 向量数据库类型，如果为None则使用配置文件中的默认类型
            batch_size: 批次处理大小，默认64
            max_concurrent_batches: 最大并发批次数量，默认5

        Returns:
            VectorDBAdapter: 向量数据库适配器实例

        Raises:
            ValueError: 不支持的数据库类型
            ImportError: 缺少必要的依赖库
        """
        if db_type is None:
            config = get_vector_db_config()
            db_type = config.db_type

        logger.info(
            f"创建向量数据库适配器: {db_type}，批次大小: {batch_size}，最大并发批次: {max_concurrent_batches}"
        )

        if db_type == VectorDBType.CHROMA:
            return ChromaAdapter(
                embeddings,
                batch_size=batch_size,
                max_concurrent_batches=max_concurrent_batches,
            )
        elif db_type == VectorDBType.MILVUS:
            return MilvusAdapter(
                embeddings,
                batch_size=batch_size,
                max_concurrent_batches=max_concurrent_batches,
            )
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")

    @staticmethod
    def create_vector_store(
        embeddings: Embeddings,
        collection_name: str,
        db_type: Optional[VectorDBType] = None,
        batch_size: int = 64,
        max_concurrent_batches: int = 5,
    ) -> VectorStoreAdapter:
        """
        创建向量存储适配器

        Args:
            embeddings: 嵌入模型实例
            collection_name: 集合名称
            db_type: 向量数据库类型，如果为None则使用配置文件中的默认类型
            batch_size: 批次处理大小，默认64
            max_concurrent_batches: 最大并发批次数量，默认5

        Returns:
            VectorStoreAdapter: 向量存储适配器实例
        """
        adapter = VectorDBFactory.create_adapter(
            embeddings, db_type, batch_size, max_concurrent_batches
        )
        return VectorStoreAdapter(adapter, collection_name)

    @staticmethod
    def get_supported_types() -> list:
        """
        获取支持的向量数据库类型列表

        Returns:
            list: 支持的数据库类型列表
        """
        return [db_type.value for db_type in VectorDBType]

    @staticmethod
    def validate_dependencies(db_type: VectorDBType) -> bool:
        """
        验证指定数据库类型的依赖是否满足

        Args:
            db_type: 向量数据库类型

        Returns:
            bool: 依赖是否满足
        """
        try:
            if db_type == VectorDBType.CHROMA:
                import langchain_chroma

                return True
            elif db_type == VectorDBType.MILVUS:
                import pymilvus

                return True
            else:
                return False
        except ImportError:
            logger.warning(f"向量数据库 {db_type} 的依赖库未安装")
            return False


class VectorDBManager:
    """向量数据库管理器，提供统一的管理接口"""

    def __init__(
        self,
        embeddings: Embeddings,
        db_type: Optional[VectorDBType] = None,
        batch_size: int = 64,
        max_concurrent_batches: int = 5,
    ):
        """
        初始化向量数据库管理器

        Args:
            embeddings: 嵌入模型实例
            db_type: 向量数据库类型
            batch_size: 批次处理大小，默认64
            max_concurrent_batches: 最大并发批次数量，默认5
        """
        self.embeddings = embeddings
        self.db_type = db_type or get_vector_db_config().db_type
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self._adapter: Optional[VectorDBAdapter] = None
        self._collections: dict = {}  # 缓存已创建的collection适配器

        # 验证依赖
        if not VectorDBFactory.validate_dependencies(self.db_type):
            raise ImportError(f"向量数据库 {self.db_type} 的依赖库未安装")

        logger.info(
            f"VectorDBManager 初始化完成，使用数据库类型: {self.db_type}，"
            f"批次大小: {batch_size}，最大并发批次: {max_concurrent_batches}"
        )

    @property
    def adapter(self) -> VectorDBAdapter:
        """获取向量数据库适配器实例（延迟初始化）"""
        if self._adapter is None:
            self._adapter = VectorDBFactory.create_adapter(
                self.embeddings,
                self.db_type,
                self.batch_size,
                self.max_concurrent_batches,
            )
        return self._adapter

    def get_collection(self, collection_name: str) -> VectorStoreAdapter:
        """
        获取集合适配器

        Args:
            collection_name: 集合名称

        Returns:
            VectorStoreAdapter: 集合适配器实例
        """
        if collection_name not in self._collections:
            self._collections[collection_name] = VectorStoreAdapter(
                self.adapter, collection_name
            )
        return self._collections[collection_name]

    async def create_collection(self, collection_name: str) -> bool:
        """
        创建集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 创建是否成功
        """
        return await self.adapter.create_collection(collection_name)

    async def collection_exists(self, collection_name: str) -> bool:
        """
        检查集合是否存在

        Args:
            collection_name: 集合名称

        Returns:
            bool: 集合是否存在
        """
        return await self.adapter.collection_exists(collection_name)

    async def delete_collection(self, collection_name: str) -> bool:
        """
        删除集合

        Args:
            collection_name: 集合名称

        Returns:
            bool: 删除是否成功
        """
        # 从缓存中移除
        if collection_name in self._collections:
            del self._collections[collection_name]

        return await self.adapter.delete_collection(collection_name)

    async def close(self):
        """关闭连接，清理资源"""
        if self._adapter:
            await self._adapter.close()
        self._collections.clear()
        logger.debug("VectorDBManager 资源已清理")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()


# 单例工厂实例，用于全局访问
_global_factory = VectorDBFactory()


def get_vector_db_factory() -> VectorDBFactory:
    """获取全局向量数据库工厂实例"""
    return _global_factory
