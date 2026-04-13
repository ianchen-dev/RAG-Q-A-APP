# 导入必要的类型提示
from typing import List, Optional, Tuple

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 导入 SemanticChunker 时处理可能的 ImportError
try:
    from langchain_experimental.text_splitter import SemanticChunker

    LANGCHAIN_EXPERIMENTAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_EXPERIMENTAL_AVAILABLE = False
    SemanticChunker = None

from unstructured.file_utils.filetype import FileType, detect_filetype

"""
Note: If using libmagic for file type detection, add locale workaround at:
unstructured/file_utils/filetype.py:361
if LIBMAGIC_AVAILABLE:
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
"""


class DocumentChunker(BaseLoader):
    """
    文档加载与切分。
    支持多种分割策略：
    - recursive: 递归字符分割，适用于一般文本
    - semantic: 语义分割，适用于需要保持语义完整性的场景
    - markdown: 基于Markdown标题结构分割，仅适用于Markdown文件
    - hybrid: 智能混合分割策略，根据文件类型自动选择最佳分割方法
    """

    # Class constant for reusable TextLoader configuration
    TEXT_LOADER_CONFIG = (TextLoader, {"autodetect_encoding": True, "encoding": "utf-8"})

    allow_file_type = {  # 文件类型与加载类及参数
        FileType.CSV: (CSVLoader, {"autodetect_encoding": True, "encoding": "utf-8"}),
        FileType.TXT: TEXT_LOADER_CONFIG,
        FileType.DOC: (UnstructuredWordDocumentLoader, {"encoding": "utf-8"}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {"encoding": "utf-8"}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {"encoding": "utf-8"}),
    }

    # 文件类型对应的默认分割策略
    default_splitting_strategy = {
        FileType.CSV: "recursive",  # CSV 文件使用递归分割
        FileType.TXT: "recursive",  # 文本文件使用递归分割
        FileType.DOC: "recursive",  # Word 文档使用递归分割
        FileType.DOCX: "recursive",  # Word 文档使用递归分割
        FileType.PDF: "recursive",  # PDF 文件使用递归分割
        FileType.MD: "markdown",  # Markdown 文件使用专门的分割器
    }

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 25,
        splitter_type: str = "hybrid",  # 'recursive', 'semantic', 'markdown', 或 'hybrid'
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """
        初始化文档分割器。

        :param file_path: 文件路径
        :param chunk_size: 分块大小（对recursive和hybrid模式有效）
        :param chunk_overlap: 分块重叠（对recursive和hybrid模式有效）
        :param splitter_type: 分割策略类型
        :param embeddings: 嵌入模型（对semantic和hybrid模式有效）
        """
        self.file_path = file_path
        try:
            self.file_type_ = detect_filetype(file_path)
        except Exception as e:
            print(f"警告: 检测文件类型时出错 '{file_path}': {e}. 默认使用 TXT 类型。")
            self.file_type_ = FileType.TXT

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings

        if self.file_type_ not in self.allow_file_type:
            print(
                f"警告: 文件类型 {self.file_type_} 未在 allow_file_type 中明确定义。将尝试使用 TextLoader。"
            )
            self.allow_file_type[self.file_type_] = self.TEXT_LOADER_CONFIG

        # 如果使用 hybrid 模式，根据文件类型选择合适的分割策略
        effective_splitter_type = splitter_type
        if splitter_type == "hybrid":
            effective_splitter_type = self.default_splitting_strategy.get(
                self.file_type_, "recursive"
            )
            print(
                f"使用文件类型 {self.file_type_} 的默认分割策略: {effective_splitter_type}"
            )
        self.splitter_type = effective_splitter_type

        # 初始化加载器 - 确定正确的加载器配置
        loader_class, params = self._get_loader_config()
        self.loader: BaseLoader = loader_class(self.file_path, **params)

        # 初始化分割器
        self._init_splitter()

    def _get_loader_config(self) -> Tuple[type, dict]:
        """获取适当的加载器类和参数。"""
        if self.file_type_ == FileType.MD and self.splitter_type == "markdown":
            print(
                "检测到 Markdown 文件和 markdown 分割策略，切换到 TextLoader 以保留原始标题进行分割。"
            )
            return self.TEXT_LOADER_CONFIG
        return self.allow_file_type[self.file_type_]

    def _init_splitter(self) -> None:
        """初始化文本分割器"""
        # 处理无效的 markdown 策略
        if self.splitter_type == "markdown" and self.file_type_ != FileType.MD:
            print(
                f"警告：markdown 分割策略通常与 Markdown 文件配合使用效果最佳，当前文件类型为 {self.file_type_}。"
            )
            print("自动切换为 recursive 分割策略。")
            self.splitter_type = "recursive"

        # 分发到相应的初始化器
        initializer = {
            "semantic": self._init_semantic_splitter,
            "markdown": self._init_markdown_splitter,
        }.get(self.splitter_type, self._init_recursive_splitter)

        initializer()

    def _init_semantic_splitter(self) -> None:
        """初始化语义分割器"""
        if not LANGCHAIN_EXPERIMENTAL_AVAILABLE:
            raise ImportError(
                "无法使用 'semantic' 分割器，因为 langchain_experimental 未安装。"
                "请运行 'uv add langchain_experimental' 或 'pip install langchain_experimental'."
            )
        if self.embeddings is None:
            raise ValueError("必须为 'semantic' 分割器提供 embeddings 参数。")

        try:
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
            )
            print("使用 SemanticChunker 进行文本分割。")
        except Exception as e:
            print(f"初始化 SemanticChunker 时出错: {e}")
            raise

    def _init_recursive_splitter(self) -> None:
        """初始化递归字符分割器"""
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
        )
        print(
            f"使用 RecursiveCharacterTextSplitter (块大小={self.chunk_size}, 重叠={self.chunk_overlap})。"
        )

    def _init_markdown_splitter(self) -> None:
        """初始化Markdown结构分割器"""
        headers_to_split_on = [
            ("#", "标题1"),
            ("##", "标题2"),
            ("###", "标题3"),
            ("####", "标题4"),
        ]
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        print("使用 MarkdownHeaderTextSplitter 进行文档结构分割。")

    def load(self) -> List[Document]:
        """加载并分割文档"""
        print(f"开始使用 '{self.splitter_type}' 分割器加载并分割文档: {self.file_path}")
        try:
            initial_docs = self.loader.load()
            if not initial_docs:
                print(f"警告：加载器未能从 {self.file_path} 加载任何文档。")
                return []

            # 根据策略分割文档
            if self.file_type_ == FileType.MD and self.splitter_type == "markdown":
                final_docs = self._split_markdown_document(initial_docs)
            else:
                final_docs = self.text_splitter.split_documents(initial_docs)

            print(f"文档分割完成，共生成 {len(final_docs)} 个块。")
            return final_docs

        except Exception as e:
            print(
                f"使用 '{self.splitter_type}' 分割器处理文档 '{self.file_path}' 时出错: {e}"
            )
            return []

    def _split_markdown_document(self, initial_docs: List[Document]) -> List[Document]:
        """分割 Markdown 文档并保留标题元数据"""
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

