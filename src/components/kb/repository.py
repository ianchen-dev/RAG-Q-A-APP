"""
知识库存储组件

封装知识库的 MongoDB 操作，提供数据持久化接口
"""

import logging
from typing import List, Optional

from bson import ObjectId

from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel

logger = logging.getLogger(__name__)


class KnowledgeBaseRepository:
    """
    知识库存储器

    负责知识库的 MongoDB 操作，包括 CRUD 和文件列表管理
    """

    def __init__(self):
        """初始化存储器"""
        self.logger = logger

    async def find_by_id(self, kb_id: str) -> Optional[KnowledgeBaseModel]:
        """
        根据 ID 查找知识库

        Args:
            kb_id: 知识库 ID 字符串

        Returns:
            找到的知识库文档，如果不存在则返回 None
        """
        self.logger.debug(f"查找知识库: {kb_id}")
        return await KnowledgeBaseModel.get(ObjectId(kb_id))

    async def find_all(self) -> List[KnowledgeBaseModel]:
        """
        获取所有知识库

        Returns:
            所有知识库文档的列表
        """
        db_knowledge_list = await KnowledgeBaseModel.find_all().to_list()
        self.logger.info(f"从 MongoDB 加载了 {len(db_knowledge_list)} 个知识库。")
        return db_knowledge_list

    async def save(self, kb: KnowledgeBaseModel) -> KnowledgeBaseModel:
        """
        保存新知识库

        Args:
            kb: 要保存的知识库文档

        Returns:
            保存后的知识库文档
        """
        await kb.insert()
        self.logger.info(f"知识库已保存: {kb.id}")
        return kb

    async def delete(self, kb_id: str) -> bool:
        """
        删除知识库

        Args:
            kb_id: 要删除的知识库 ID

        Returns:
            如果删除成功则为 True
        """
        kb_doc = await self.find_by_id(kb_id)
        if not kb_doc:
            return False

        delete_result = await kb_doc.delete()
        if not delete_result:
            self.logger.warning(f"MongoDB 记录 '{kb_id}' 可能未删除成功。")
            return False

        self.logger.info(f"MongoDB 记录 '{kb_id}' 删除成功。")
        return True

    async def add_file_to_list(self, kb_id: str, file_metadata: dict) -> None:
        """
        添加文件到知识库的文件列表

        Args:
            kb_id: 知识库 ID
            file_metadata: 要添加的文件元数据字典
        """
        kb_doc = await self.find_by_id(kb_id)
        if not kb_doc:
            raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

        await kb_doc.update({"$push": {"filesList": file_metadata}})
        self.logger.info(
            f"文件 {file_metadata.get('file_name')} "
            f"(MD5: {file_metadata.get('file_md5')}) "
            f"元数据已添加到 MongoDB 知识库 {kb_id}。"
        )

    async def remove_file_from_list(self, kb_id: str, file_md5: str) -> None:
        """
        从知识库的文件列表中移除文件

        Args:
            kb_id: 知识库 ID
            file_md5: 要移除的文件 MD5
        """
        kb_doc = await self.find_by_id(kb_id)
        if not kb_doc:
            raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

        self.logger.info(
            f"从 MongoDB 知识库 {kb_id} 的 filesList 中移除 MD5: {file_md5}"
        )
        await kb_doc.update({"$pull": {"filesList": {"file_md5": file_md5}}})
