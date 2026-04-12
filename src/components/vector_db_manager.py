"""
向量数据库管理组件

负责向量数据库的删除和集合管理操作
"""

import logging
from typing import Any, Dict

from src.components.kb_factory import KnowledgeBaseFactory
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    向量数据库管理器

    负责向量数据库的集合管理操作
    """

    def __init__(self, kb_factory: KnowledgeBaseFactory):
        """
        初始化向量数据库管理器

        Args:
            kb_factory: 知识库工厂，用于创建 Knowledge 实例
        """
        self.kb_factory = kb_factory
        self.logger = logger

    async def delete_collection(self, kb_id: str, kb_doc: KnowledgeBaseModel) -> bool:
        """
        删除向量数据库集合

        Args:
            kb_id: 知识库 ID
            kb_doc: 知识库文档对象

        Returns:
            如果删除成功则为 True，否则为 False
        """
        kb_id_str = str(kb_id)
        config = kb_doc.embedding_config

        if not (config and config.embedding_supplier and config.embedding_model):
            self.logger.warning(
                f"知识库 {kb_id} 缺少嵌入配置，跳过向量数据库集合删除。"
            )
            return False

        try:
            knowledge_util = self.kb_factory.create_knowledge_instance(kb_doc)
            collection_deleted = await knowledge_util.delete_collection(kb_id_str)

            if collection_deleted:
                self.logger.info(f"向量数据库集合 '{kb_id_str}' 删除成功。")
            else:
                self.logger.warning(f"向量数据库集合 '{kb_id_str}' 删除失败或不存在。")

            await knowledge_util.close()
            return collection_deleted

        except Exception as e:
            self.logger.error(f"删除向量数据库集合 '{kb_id_str}' 时出错: {e}")
            return False

    async def delete_vectors_by_filter(
        self, kb_id: str, filter_dict: Dict[str, Any], kb_doc: KnowledgeBaseModel
    ) -> bool:
        """
        根据条件删除向量

        Args:
            kb_id: 知识库 ID
            filter_dict: 用于过滤的字典
            kb_doc: 知识库文档对象

        Returns:
            如果删除成功则为 True，否则为 False
        """
        kb_id_str = str(kb_id)
        knowledge_util = self.kb_factory.create_knowledge_instance(kb_doc)

        try:
            collection_exists = await knowledge_util.collection_exists(kb_id_str)

            if not collection_exists:
                self.logger.info(f"向量数据库集合 '{kb_id_str}' 不存在，无需删除向量。")
                return True

            self.logger.info(
                f"准备从向量数据库集合 '{kb_id_str}' 删除满足条件的向量..."
            )

            deletion_success = await knowledge_util.delete_documents_by_filter(
                kb_id_str, filter_dict
            )

            if deletion_success:
                self.logger.info(
                    f"向量数据库集合 '{kb_id_str}' 中满足条件的向量已删除。"
                )
            else:
                self.logger.warning(
                    f"从向量数据库集合 '{kb_id_str}' 删除向量时未成功。"
                )

            return deletion_success

        finally:
            await knowledge_util.close()

    async def collection_exists(self, kb_id: str, kb_doc: KnowledgeBaseModel) -> bool:
        """
        检查向量数据库集合是否存在

        Args:
            kb_id: 知识库 ID
            kb_doc: 知识库文档对象

        Returns:
            如果集合存在则为 True，否则为 False
        """
        kb_id_str = str(kb_id)
        knowledge_util = self.kb_factory.create_knowledge_instance(kb_doc)

        try:
            return await knowledge_util.collection_exists(kb_id_str)
        finally:
            await knowledge_util.close()
