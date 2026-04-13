"""Text splitter factory component for RAG ChatBot.

This module provides a factory for creating appropriate LangChain text splitters
based on the requested splitting strategy. It supports multiple strategies:
- recursive: Recursive character splitting for general text
- semantic: Semantic chunking for maintaining semantic coherence
- markdown: Markdown structure-aware splitting
"""

import logging
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Import SemanticChunker with graceful error handling
try:
    from langchain_experimental.text_splitter import SemanticChunker

    LANGCHAIN_EXPERIMENTAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_EXPERIMENTAL_AVAILABLE = False
    SemanticChunker = None

logger = logging.getLogger(__name__)

# Supported splitter types
SplitterType = Literal["recursive", "semantic", "markdown"]


class TextSplitterFactory:
    """Factory for creating text splitters based on strategy.

    This factory encapsulates the complexity of initializing different
    text splitters with their appropriate configurations.
    """

    # Default separators for recursive splitting (Chinese punctuation aware)
    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        " ",
        "",
    ]

    # Default headers for markdown splitting
    DEFAULT_MARKDOWN_HEADERS = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    @classmethod
    def create_recursive_splitter(
        cls,
        chunk_size: int = 500,
        chunk_overlap: int = 25,
        separators: list | None = None,
    ) -> RecursiveCharacterTextSplitter:
        """Create a recursive character text splitter.

        This splitter recursively splits text while trying to keep
        related pieces together using natural separators.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks for context continuity
            separators: List of separators to try (in order).
                If None, uses default Chinese-aware separators.

        Returns:
            RecursiveCharacterTextSplitter: Configured splitter instance

        Example:
            >>> splitter = TextSplitterFactory.create_recursive_splitter(
            ...     chunk_size=1000, chunk_overlap=100
            ... )
        """
        if separators is None:
            separators = cls.DEFAULT_SEPARATORS

        logger.debug(
            f"创建 RecursiveCharacterTextSplitter: "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

    @classmethod
    def create_semantic_splitter(
        cls,
        embeddings: Embeddings,
        breakpoint_threshold_type: str = "percentile",
    ) -> "SemanticChunker":
        """Create a semantic text splitter.

        This splitter uses embeddings to split text at semantic boundaries,
        keeping related content together.

        Args:
            embeddings: Embedding model for semantic analysis
            breakpoint_threshold_type: Method for determining split boundaries.
                Options: 'percentile', 'standard_deviation', 'interquartile'

        Returns:
            SemanticChunker: Configured splitter instance

        Raises:
            ImportError: If langchain_experimental is not installed
            ValueError: If embeddings is None

        Example:
            >>> from src.components.embedding_provider import create_embedding
            >>> embeddings = create_embedding("ollama", "nomic-embed-text")
            >>> splitter = TextSplitterFactory.create_semantic_splitter(
            ...     embeddings
            ... )
        """
        if not LANGCHAIN_EXPERIMENTAL_AVAILABLE:
            raise ImportError(
                "无法使用 'semantic' 分割器，因为 langchain_experimental 未安装。"
                "请运行 'uv sync --group semantic' 或 "
                "'pip install langchain_experimental'。"
            )

        if embeddings is None:
            raise ValueError("必须为 'semantic' 分割器提供 embeddings 参数。")

        logger.debug(
            f"创建 SemanticChunker: "
            f"breakpoint_threshold_type={breakpoint_threshold_type}"
        )

        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    @classmethod
    def create_markdown_splitter(
        cls,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> MarkdownHeaderTextSplitter:
        """Create a markdown structure-aware text splitter.

        This splitter splits markdown documents based on their header structure,
        preserving the hierarchy in metadata.

        Args:
            headers_to_split_on: List of (header, name) tuples.
                If None, uses default headers (# through ####).

        Returns:
            MarkdownHeaderTextSplitter: Configured splitter instance

        Example:
            >>> splitter = TextSplitterFactory.create_markdown_splitter()
            >>> splits = splitter.split_text(markdown_content)
        """
        if headers_to_split_on is None:
            headers_to_split_on = cls.DEFAULT_MARKDOWN_HEADERS

        logger.debug(
            f"创建 MarkdownHeaderTextSplitter: "
            f"headers={[h[0] for h in headers_to_split_on]}"
        )

        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

    @classmethod
    def create_splitter(
        cls,
        splitter_type: SplitterType,
        chunk_size: int = 500,
        chunk_overlap: int = 25,
        embeddings: Embeddings | None = None,
    ):
        """Create a text splitter based on the specified type.

        This is a convenience method that routes to the appropriate
        splitter creation method.

        Args:
            splitter_type: Type of splitter to create
            chunk_size: Chunk size for recursive splitter
            chunk_overlap: Overlap for recursive splitter
            embeddings: Required for semantic splitter

        Returns:
            Configured splitter instance

        Raises:
            ValueError: If splitter_type is invalid or requirements not met

        Example:
            >>> splitter = TextSplitterFactory.create_splitter(
            ...     "recursive", chunk_size=1000
            ... )
        """
        logger.info(f"创建文本分割器: 类型={splitter_type}")

        if splitter_type == "recursive":
            return cls.create_recursive_splitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif splitter_type == "semantic":
            return cls.create_semantic_splitter(
                embeddings=embeddings,  # type: ignore
            )
        elif splitter_type == "markdown":
            return cls.create_markdown_splitter()
        else:
            raise ValueError(
                f"不支持的分割器类型: {splitter_type}. "
                f"支持的类型: 'recursive', 'semantic', 'markdown'"
            )

    @classmethod
    def is_semantic_available(cls) -> bool:
        """Check if semantic chunking is available.

        Returns:
            True if langchain_experimental is installed
        """
        return LANGCHAIN_EXPERIMENTAL_AVAILABLE
