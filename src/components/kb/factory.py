"""
知识库工厂组件

负责创建 KnowledgeManager 实例和知识库对象
"""

import logging
from typing import Optional

from src.components.kb.knowledge_manager import KnowledgeManager
from src.components.kb.repository import KnowledgeBaseRepository
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel
from src.utils.embedding import get_embedding

logger = logging.getLogger(__name__)

# 批处理默认值
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CONCURRENT_BATCHES = 5


class KnowledgeBaseFactory:
    """
    知识库工厂

    负责创建 KnowledgeManager 工具实例和新的知识库记录
    """

    def __init__(
        self,
        default_batch_size: int = DEFAULT_BATCH_SIZE,
        default_max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_BATCHES,
    ):
        """
        初始化工厂

        Args:
            default_batch_size: 默认嵌入批次大小
            default_max_concurrent_batches: 默认最大并发批次数
        """
        self.default_batch_size = default_batch_size
        self.default_max_concurrent_batches = default_max_concurrent_batches
        self.logger = logger

    def create_knowledge_manager(
        self,
        kb_doc: KnowledgeBaseModel,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
    ) -> KnowledgeManager:
        """
        使用给定知识库的嵌入配置创建 KnowledgeManager 工具实例

        Args:
            kb_doc: 包含嵌入配置的知识库文档
            batch_size: 嵌入批次大小（如果为 None 则使用默认值）
            max_concurrent_batches: 最大并发批次数（如果为 None 则使用默认值）

        Returns:
            配置好的 KnowledgeManager 实例
        """
        config = kb_doc.embedding_config
        self.logger.info(
            f"使用知识库的嵌入配置: "
            f"supplier='{config.embedding_supplier}', "
            f"model='{config.embedding_model}'"
        )

        _embedding = get_embedding(
            config.embedding_supplier,
            config.embedding_model,
            config.embedding_apikey,
        )

        return KnowledgeManager(
            _embeddings=_embedding,
            vector_db_type=None,
            splitter="hybrid",
            use_bm25=False,
            use_reranker=False,
            batch_size=batch_size or self.default_batch_size,
            max_concurrent_batches=max_concurrent_batches
            or self.default_max_concurrent_batches,
        )

    def create_knowledge_base(
        self, knowledge_base_data, current_user
    ) -> KnowledgeBaseModel:
        """
        创建新的知识库记录

        Args:
            knowledge_base_data: 知识库创建数据
            current_user: 当前创建者用户

        Returns:
            创建的 KnowledgeBaseModel 实例
        """
        new_knowledge_base = KnowledgeBaseModel(
            title=knowledge_base_data.title,
            tag=knowledge_base_data.tag,
            description=knowledge_base_data.description,
            creator=current_user.username,
            embedding_config=knowledge_base_data.embedding_config,
            filesList=[],
        )

        self.logger.info(
            f"创建新知识库记录: title='{knowledge_base_data.title}', "
            f"creator='{current_user.username}'"
        )

        return new_knowledge_base

    def create_repository(self) -> KnowledgeBaseRepository:
        """
        创建知识库存储器实例

        Returns:
            KnowledgeBaseRepository 实例
        """
        return KnowledgeBaseRepository()
