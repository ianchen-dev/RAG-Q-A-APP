"""Document loader factory component for RAG ChatBot.

This module provides a factory for creating appropriate LangChain document loaders
based on file type detection. It encapsulates the complexity of mapping file types
to their corresponding loaders and configuration parameters.
"""

import logging
from typing import Dict, Tuple, Type

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.document_loaders import BaseLoader
from unstructured.file_utils.filetype import FileType

logger = logging.getLogger(__name__)


class DocumentLoaderConfig:
    """Configuration for document loaders.

    Attributes:
        loader_class: The LangChain loader class to use
        params: Dictionary of parameters to pass to the loader constructor
    """

    def __init__(self, loader_class: Type[BaseLoader], params: Dict[str, any]):
        """Initialize loader configuration.

        Args:
            loader_class: The LangChain loader class
            params: Parameters for the loader constructor
        """
        self.loader_class = loader_class
        self.params = params


class DocumentLoaderFactory:
    """Factory for creating document loaders based on file type.

    This factory provides a centralized way to map file types to their
    appropriate LangChain document loaders with proper configuration.
    """

    # Class constant for reusable TextLoader configuration
    TEXT_LOADER_CONFIG = DocumentLoaderConfig(
        TextLoader,
        {"autodetect_encoding": True, "encoding": "utf-8"},
    )

    # File type mapping to loader configurations
    LOADER_mappings: Dict[FileType, DocumentLoaderConfig] = {
        FileType.CSV: DocumentLoaderConfig(
            CSVLoader,
            {"autodetect_encoding": True, "encoding": "utf-8"},
        ),
        FileType.TXT: TEXT_LOADER_CONFIG,
        FileType.DOC: DocumentLoaderConfig(
            UnstructuredWordDocumentLoader,
            {"encoding": "utf-8"},
        ),
        FileType.DOCX: DocumentLoaderConfig(
            UnstructuredWordDocumentLoader,
            {"encoding": "utf-8"},
        ),
        FileType.PDF: DocumentLoaderConfig(
            PyPDFLoader,
            {},
        ),
        FileType.MD: DocumentLoaderConfig(
            UnstructuredMarkdownLoader,
            {"encoding": "utf-8"},
        ),
    }

    @classmethod
    def get_loader_config(cls, file_type: FileType) -> DocumentLoaderConfig:
        """Get the loader configuration for a given file type.

        Args:
            file_type: The detected file type

        Returns:
            DocumentLoaderConfig: Configuration for the appropriate loader

        Example:
            >>> config = DocumentLoaderFactory.get_loader_config(FileType.PDF)
            >>> loader = config.loader_class("path/to/file.pdf", **config.params)
        """
        if file_type not in cls.LOADER_mappings:
            logger.warning(
                f"文件类型 {file_type} 未明确定义，将使用 TextLoader 作为默认配置。"
            )
            return cls.TEXT_LOADER_CONFIG

        return cls.LOADER_mappings[file_type]

    @classmethod
    def create_loader(
        cls,
        file_path: str,
        file_type: FileType,
        force_text_loader: bool = False,
    ) -> BaseLoader:
        """Create a document loader instance for the given file.

        Args:
            file_path: Path to the file to load
            file_type: The detected file type
            force_text_loader: If True, use TextLoader regardless of file type.
                Useful for markdown files when using custom splitters.

        Returns:
            BaseLoader: An instantiated document loader

        Example:
            >>> loader = DocumentLoaderFactory.create_loader(
            ...     "doc.pdf", FileType.PDF
            ... )
            >>> docs = loader.load()
        """
        if force_text_loader:
            logger.debug(
                "强制使用 TextLoader 以保留原始内容供自定义分割器处理。"
            )
            config = cls.TEXT_LOADER_CONFIG
        else:
            config = cls.get_loader_config(file_type)

        logger.debug(
            f"为文件 '{file_path}' (类型: {file_type}) 创建加载器: "
            f"{config.loader_class.__name__}"
        )
        return config.loader_class(file_path, **config.params)

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported file types.

        Returns:
            List of supported FileType enum values
        """
        return list(cls.LOADER_mappings.keys())
