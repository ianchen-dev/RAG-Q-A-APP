"""
知识库验证组件

提供知识库配置和数据的验证功能，包括：
- 嵌入配置验证
- 文件重复检查
- 知识库 ID 格式验证
"""

import logging

from bson import ObjectId

from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel

logger = logging.getLogger(__name__)


class KnowledgeBaseValidator:
    """
    知识库验证器

    负责验证知识库相关的配置和数据完整性
    """

    def __init__(self):
        """初始化验证器"""
        self.logger = logger

    def validate_embedding_config(self, kb_doc: KnowledgeBaseModel, kb_id: str) -> None:
        """
        验证知识库是否具有完整的嵌入配置

        Args:
            kb_doc: 要验证的知识库文档
            kb_id: 知识库 ID，用于错误消息

        Raises:
            ValueError: 如果嵌入配置缺失或不完整
        """
        if not kb_doc.embedding_config:
            raise ValueError(f"知识库 {kb_id} 缺少嵌入配置 (embedding_config)。")

        if (
            not kb_doc.embedding_config.embedding_model
            or not kb_doc.embedding_config.embedding_supplier
        ):
            raise ValueError(f"知识库 {kb_id} 的嵌入配置不完整。")

        self.logger.debug(
            f"知识库 {kb_id} 嵌入配置验证通过: "
            f"supplier={kb_doc.embedding_config.embedding_supplier}, "
            f"model={kb_doc.embedding_config.embedding_model}"
        )

    def check_file_duplicate(
        self,
        kb_doc: KnowledgeBaseModel,
        kb_id: str,
        file_md5: str,
        file_name: str,
    ) -> None:
        """
        检查具有给定 MD5 的文件是否已存在于知识库中

        Args:
            kb_doc: 知识库文档
            kb_id: 知识库 ID
            file_md5: 文件的 MD5 哈希值
            file_name: 文件名

        Raises:
            ValueError: 如果文件已存在
        """
        if kb_doc.filesList:
            for existing_file in kb_doc.filesList:
                if existing_file.get("file_md5") == file_md5:
                    self.logger.warning(
                        f"文件 (MD5: {file_md5}) 已存在于知识库 {kb_id}，跳过处理。"
                    )
                    raise ValueError(
                        f"文件 '{file_name}' (MD5: {file_md5}) 已存在于此知识库。"
                    )

        self.logger.debug(f"文件 {file_name} (MD5: {file_md5}) 重复检查通过")

    def validate_kb_id(self, kb_id: str) -> bool:
        """
        验证知识库 ID 格式是否有效

        Args:
            kb_id: 要验证的知识库 ID

        Returns:
            如果 ID 格式有效则为 True

        Raises:
            ValueError: 如果 ID 格式无效
        """
        if not ObjectId.is_valid(kb_id):
            raise ValueError(f"无效的知识库 ID 格式: {kb_id}")

        self.logger.debug(f"知识库 ID {kb_id} 格式验证通过")
        return True

    def validate_file_exists(self, file_path: str) -> None:
        """
        验证文件是否存在

        Args:
            file_path: 要验证的文件路径

        Raises:
            FileNotFoundError: 如果文件不存在
        """
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        self.logger.debug(f"文件存在性验证通过: {file_path}")
