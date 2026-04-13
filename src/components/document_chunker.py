"""Document chunker component for RAG ChatBot.

This module provides a unified component for document loading and chunking.
It coordinates the document loader factory and text splitter factory to provide
a simple interface for processing documents with various splitting strategies.

The component supports multiple splitting strategies:
- recursive: Recursive character splitting, suitable for general text
- semantic: Semantic splitting that maintains semantic coherence
- markdown: Markdown structure-based splitting
- hybrid: Automatically selects the best strategy based on file type
"""

import logging
from typing import List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from unstructured.file_utils.filetype import FileType, detect_filetype

from src.components.document_loader_factory import DocumentLoaderFactory
from src.components.text_splitter_factory import SplitterType, TextSplitterFactory

logger = logging.getLogger(__name__)

# All supported splitter types
AllSplitterType = Literal["recursive", "semantic", "markdown", "hybrid"]


class DocumentChunkerConfig:
    """Configuration for document chunking.

    Attributes:
        chunk_size: Maximum size of each chunk (for recursive/hybrid)
        chunk_overlap: Overlap between chunks (for recursive/hybrid)
        splitter_type: Strategy for splitting documents
        embeddings: Embedding model (required for semantic/hybrid with markdown)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 25,
        splitter_type: AllSplitterType = "hybrid",
        embeddings: Optional[Embeddings] = None,
    ):
        """Initialize chunker configuration.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            splitter_type: Splitting strategy to use
            embeddings: Embedding model (required for semantic)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
        self.embeddings = embeddings

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.splitter_type == "semantic" and self.embeddings is None:
            raise ValueError("必须为 'semantic' 分割器提供 embeddings 参数。")


class DocumentChunker:
    """
    文档加载与切分组件。

    支持多种分割策略：
    - recursive: 递归字符分割，适用于一般文本
    - semantic: 语义分割，适用于需要保持语义完整性的场景
    - markdown: 基于 Markdown 标题结构分割，仅适用于 Markdown 文件
    - hybrid: 智能混合分割策略，根据文件类型自动选择最佳分割方法

    This component coordinates the document loading and splitting process,
    delegating the actual work to specialized factories.
    """

    # Default splitting strategy for each file type
    DEFAULT_SPLITTING_STRATEGY = {
        FileType.CSV: "recursive",
        FileType.TXT: "recursive",
        FileType.DOC: "recursive",
        FileType.DOCX: "recursive",
        FileType.PDF: "recursive",
        FileType.MD: "markdown",
    }

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 25,
        splitter_type: AllSplitterType = "hybrid",
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """初始化文档分割器。

        Args:
            file_path: 文件路径
            chunk_size: 分块大小（对 recursive 和 hybrid 模式有效）
            chunk_overlap: 分块重叠（对 recursive 和 hybrid 模式有效）
            splitter_type: 分割策略类型
            embeddings: 嵌入模型（对 semantic 和 hybrid 模式有效）

        Raises:
            ValueError: If configuration is invalid
        """
        self.file_path = file_path

        # Detect file type
        try:
            self.file_type = detect_filetype(file_path)
        except Exception as e:
            logger.warning(
                f"检测文件类型时出错 '{file_path}': {e}. 默认使用 TXT 类型。"
            )
            self.file_type = FileType.TXT

        # Create configuration
        self.config = DocumentChunkerConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type,
            embeddings=embeddings,
        )
        self.config.validate()

        # Determine effective splitter type (handle hybrid mode)
        self.effective_splitter_type = self._resolve_splitter_type()

        # Initialize loader and splitter
        self.loader = DocumentLoaderFactory.create_loader(
            file_path,
            self.file_type,
            force_text_loader=self._should_use_text_loader(),
        )
        self.text_splitter = TextSplitterFactory.create_splitter(
            self.effective_splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embeddings=embeddings,
        )

        logger.info(
            f"DocumentChunker 初始化完成: "
            f"文件='{file_path}', 类型={self.file_type}, "
            f"分割器={self.effective_splitter_type}"
        )

    def _resolve_splitter_type(self) -> SplitterType:
        """Resolve the effective splitter type (handles hybrid mode).

        Returns:
            The actual splitter type to use
        """
        splitter_type = self.config.splitter_type

        if splitter_type == "hybrid":
            effective_type = self.DEFAULT_SPLITTING_STRATEGY.get(
                self.file_type, "recursive"
            )
            logger.info(
                f"使用文件类型 {self.file_type} 的默认分割策略: {effective_type}"
            )
            return effective_type  # type: ignore

        return splitter_type  # type: ignore

    def _should_use_text_loader(self) -> bool:
        """Determine if TextLoader should be forced.

        Returns True if using markdown splitter for markdown files.

        Returns:
            True if TextLoader should be used
        """
        return (
            self.file_type == FileType.MD
            and self.effective_splitter_type == "markdown"
        )

    def load(self) -> List[Document]:
        """加载并分割文档。

        Returns:
            List of split Document objects. Returns empty list on error.

        Example:
            >>> chunker = DocumentChunker("doc.pdf", splitter_type="recursive")
            >>> documents = chunker.load()
            >>> print(f"Generated {len(documents)} chunks")
        """
        logger.info(
            f"开始使用 '{self.effective_splitter_type}' 分割器 "
            f"加载并分割文档: {self.file_path}"
        )

        try:
            # Load documents
            initial_docs = self.loader.load()
            if not initial_docs:
                logger.warning(f"加载器未能从 {self.file_path} 加载任何文档。")
                return []

            # Split documents
            if (
                self.file_type == FileType.MD
                and self.effective_splitter_type == "markdown"
            ):
                final_docs = self._split_markdown_document(initial_docs)
            else:
                final_docs = self.text_splitter.split_documents(initial_docs)

            logger.info(f"文档分割完成，共生成 {len(final_docs)} 个块。")
            return final_docs

        except Exception as e:
            logger.error(
                f"使用 '{self.effective_splitter_type}' 分割器 "
                f"处理文档 '{self.file_path}' 时出错: {e}",
                exc_info=True,
            )
            return []

    def _split_markdown_document(self, initial_docs: List[Document]) -> List[Document]:
        """分割 Markdown 文档并保留标题元数据。

        Args:
            initial_docs: Initially loaded documents

        Returns:
            Split documents with preserved metadata
        """
        text = initial_docs[0].page_content
        splits = self.text_splitter.split_text(text)

        base_metadata = initial_docs[0].metadata.copy() if initial_docs else {}

        return [
            Document(
                page_content=split_doc.page_content,
                metadata={**base_metadata, **split_doc.metadata},
            )
            for split_doc in splits
        ]
