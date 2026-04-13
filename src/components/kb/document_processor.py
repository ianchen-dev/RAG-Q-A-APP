"""
知识库文档处理组件

负责文档的加载、分块、元数据注入和向量化
"""

import logging
from typing import List

from langchain_core.documents import Document

from src.utils.DocumentChunker import DocumentChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器

    负责文档的加载、分块和元数据注入
    """

    def __init__(
        self,
        splitter: str = "hybrid",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        初始化文档处理器

        Args:
            splitter: 文档分割器类型
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
        """
        self.splitter = splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger

    def load_and_chunk_documents(
        self,
        file_path: str,
        embeddings,
    ) -> List[Document]:
        """
        加载并分块文档

        Args:
            file_path: 文件路径
            embeddings: 嵌入模型实例

        Returns:
            分块后的文档列表

        Raises:
            Exception: 加载或分块失败
        """
        try:
            self.logger.debug(
                f"使用 DocumentChunker (类型: {self.splitter}) 加载和分块: {file_path}"
            )
            loader = DocumentChunker(
                file_path,
                splitter_type=self.splitter,
                embeddings=embeddings,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            documents = loader.load()
            if not documents:
                self.logger.warning(
                    f"警告: 文件 {file_path} 未产生任何文档块，跳过处理。"
                )
                return []
            self.logger.info(
                f"文件 {file_path} 加载并分块完成，共 {len(documents)} 块。"
            )
            return documents

        except Exception as e:
            self.logger.error(f"加载或分块文件 {file_path} 时出错: {e}", exc_info=True)
            raise

    def inject_metadata(
        self,
        documents: List[Document],
        kb_id: str,
        file_md5: str,
        file_name: str,
    ) -> List[Document]:
        """
        为文档注入元数据

        Args:
            documents: 文档列表
            kb_id: 知识库 ID
            file_md5: 文件 MD5
            file_name: 文件名

        Returns:
            注入元数据后的文档列表
        """
        metadata_to_add = {
            "knowledge_base_id": str(kb_id),
            "source_file_md5": file_md5,
            "source_file_name": file_name,
        }
        self.logger.debug(f"为文档块添加元数据: {metadata_to_add}")

        processed_documents = []
        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}
            current_metadata = doc.metadata.copy()
            current_metadata.update(metadata_to_add)
            processed_documents.append(
                Document(page_content=doc.page_content, metadata=current_metadata)
            )

        return processed_documents

    async def add_documents_to_collection(
        self,
        collection,
        kb_id: str,
        documents: List[Document],
        collection_exists: bool,
    ) -> None:
        """
        将文档添加到向量数据库集合

        Args:
            collection: 向量数据库集合
            kb_id: 知识库 ID
            documents: 文档列表
            collection_exists: 集合是否已存在

        Raises:
            Exception: 添加文档失败
        """
        try:
            if not collection_exists:
                self.logger.info(f"集合 '{kb_id}' 不存在，首次创建并添加文档...")
            else:
                self.logger.info(f"集合 '{kb_id}' 已存在，加载并添加新文档...")

            await collection.aadd_documents(documents)
            self.logger.info(f"文档向量数据成功添加到集合 '{kb_id}'。")

        except Exception as e:
            self.logger.error(
                f"将文档向量数据添加到集合 '{kb_id}' 时出错: {e}",
                exc_info=True,
            )
            raise
