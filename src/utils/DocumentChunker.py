# 导入必要的类型提示
from typing import List, Optional

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

# from langchain.document_loaders import
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
    SemanticChunker = None  # 定义一个占位符以便类型检查

from unstructured.file_utils.filetype import FileType, detect_filetype

"""
detect_filetype 函数中的 361行加上以下代码
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

    allow_file_type = {  # 文件类型与加载类及参数
        FileType.CSV: (CSVLoader, {"autodetect_encoding": True, "encoding": "utf-8"}),
        FileType.TXT: (TextLoader, {"autodetect_encoding": True, "encoding": "utf-8"}),
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
        chunk_size: int = 400,
        chunk_overlap: int = 20,
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
        try:  # 添加 try-except 块来处理 detect_filetype 可能的错误
            self.file_type_ = detect_filetype(file_path)
        except Exception as e:
            print(f"警告: 检测文件类型时出错 '{file_path}': {e}. 默认使用 TXT 类型。")
            self.file_type_ = FileType.TXT  # 发生错误时回退到 TXT 类型

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings

        if self.file_type_ not in self.allow_file_type:
            # 对于未明确支持的类型，可以考虑默认使用 TextLoader 或抛出错误
            print(
                f"警告: 文件类型 {self.file_type_} 未在 allow_file_type 中明确定义。将尝试使用 TextLoader。"
            )
            self.allow_file_type[self.file_type_] = (
                TextLoader,
                {"autodetect_encoding": True, "encoding": "utf-8"},
            )
            # raise ValueError(f"不支持的文件类型: {self.file_type_}") # 或者选择抛出错误

        # 如果使用 hybrid 模式，根据文件类型选择合适的分割策略
        effective_splitter_type = splitter_type
        if splitter_type == "hybrid":
            effective_splitter_type = self.default_splitting_strategy.get(
                self.file_type_, "recursive"
            )  # 提供默认值 'recursive'
            print(
                f"使用文件类型 {self.file_type_} 的默认分割策略: {effective_splitter_type}"
            )
        self.splitter_type = effective_splitter_type  # 现在设置最终的 splitter_type

        # 初始化加载器 - 先根据文件类型获取默认加载器
        loader_class, params = self.allow_file_type[self.file_type_]
        self.loader: BaseLoader = loader_class(self.file_path, **params)

        # --- 开始修改 ---
        # 特殊处理：如果文件是Markdown且使用markdown分割策略，则强制使用TextLoader
        if self.file_type_ == FileType.MD and self.splitter_type == "markdown":
            print(
                "检测到 Markdown 文件和 markdown 分割策略，切换到 TextLoader 以保留原始标题进行分割。"
            )
            # 获取 TextLoader 的参数，如果 TXT 未定义则提供默认值
            text_loader_class, text_loader_params = self.allow_file_type.get(
                FileType.TXT,
                (TextLoader, {"autodetect_encoding": True, "encoding": "utf-8"}),
            )
            self.loader = text_loader_class(self.file_path, **text_loader_params)
        # --- 结束修改 ---

        # 初始化分割器
        self._init_splitter()

    def _init_splitter(self) -> None:
        """初始化文本分割器"""
        if self.splitter_type == "semantic":
            self._init_semantic_splitter()
        elif self.splitter_type == "markdown" and self.file_type_ == FileType.MD:
            self._init_markdown_splitter()
        elif self.splitter_type == "markdown" and self.file_type_ != FileType.MD:
            print(
                f"警告：markdown 分割策略通常与 Markdown 文件配合使用 TextLoader 效果最佳，当前文件类型为 {self.file_type_}。"
            )
            # 这里可以选择是强制切换到 recursive，还是继续尝试（取决于加载器是否保留了结构）
            # 为了安全起见，如果不是MD文件却选了markdown策略，切换回recursive似乎更稳妥
            print("自动切换为 recursive 分割策略。")
            self.splitter_type = "recursive"  # 更新实例的 splitter_type
            self._init_recursive_splitter()
        else:  # default to recursive or handle other cases
            self._init_recursive_splitter()

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
            # 首先加载文档
            initial_docs = self.loader.load()
            if not initial_docs:
                print(f"警告：加载器未能从 {self.file_path} 加载任何文档。")
                return []  # 如果加载器没有返回任何文档，则提前返回空列表

            # 如果是 Markdown 文件且使用 markdown 分割策略 (此时 loader 必然是 TextLoader)
            if self.file_type_ == FileType.MD and self.splitter_type == "markdown":
                # --- 开始修改 ---
                # TextLoader 通常将整个文件加载到第一个文档的 page_content 中
                text = initial_docs[0].page_content
                # 使用 MarkdownHeaderTextSplitter 分割文本，它返回 Document 列表
                # 每个返回的 Document 包含 page_content 和与该块相关的 header metadata
                splits: List[Document] = self.text_splitter.split_text(text)

                # 准备基础元数据（来自加载器，主要是文件路径等）
                base_metadata = (
                    initial_docs[0].metadata.copy()
                    if initial_docs and initial_docs[0].metadata
                    else {}
                )

                final_docs = []
                for split_doc in splits:
                    # 创建新的元数据字典，先复制基础元数据
                    combined_metadata = base_metadata.copy()
                    # 然后更新（或添加）由 MarkdownHeaderTextSplitter 生成的特定于块的元数据（如标题）
                    combined_metadata.update(split_doc.metadata)
                    # 创建最终的 Document 对象
                    final_docs.append(
                        Document(
                            page_content=split_doc.page_content,
                            metadata=combined_metadata,
                        )
                    )
                # --- 结束修改 ---
                print(f"Markdown 文档分割完成，共生成 {len(final_docs)} 个块。")
                return final_docs
            else:
                final_docs = self.text_splitter.split_documents(initial_docs)
                print(f"文档分割完成，共生成 {len(final_docs)} 个块。")
                return final_docs

        except Exception as e:
            print(
                f"使用 '{self.splitter_type}' 分割器处理文档 '{self.file_path}' 时出错: {e}"
            )

            return []  # 或者返回空列表表示处理失败但程序继续


# if __name__ == "__main__":
#     # 示例: 使用 RecursiveCharacterTextSplitter (需要提供一个有效的docx文件路径)
#     # try:
#     #     file_path_docx = "./人事管理流程.docx"
#     #     chunker_recursive = DocumentChunker(file_path_docx, splitter_type="recursive")
#     #     chunks_recursive = chunker_recursive.load()
#     #     print(f"Recursive 分割完成，块数: {len(chunks_recursive)}")
#     #     if chunks_recursive:
#     #         print("第一个块内容预览:", chunks_recursive[0].page_content[:200])
#     # except Exception as e:
#     #     print(f"Recursive 测试失败: {e}")
#
#     # 示例: 使用 SemanticChunker (需要提供 embedding 函数和可能的依赖)
#     # 需要先安装: uv add langchain_experimental sentence-transformers bert_score
#     # try:
#     #     from langchain_community.embeddings import HuggingFaceEmbeddings
#     #     # 替换为你实际使用的 embedding 模型路径或名称
#     #     embedding_model_name = "shibing624/text2vec-base-chinese"
#     #     embeddings_instance = HuggingFaceEmbeddings(model_name=embedding_model_name)
#     #
#     #     file_path_txt = "./sample.txt" # 准备一个示例文本文件
#     #     with open(file_path_txt, "w", encoding="utf-8") as f:
#     #         f.write("这是第一个句子。这是第二个句子，它们语义上相关。\n\n这是第三个句子，与前面关系不大。这是第四个句子。")
#     #
#     #     chunker_semantic = DocumentChunker(file_path_txt, splitter_type="semantic", embeddings=embeddings_instance)
#     #     chunks_semantic = chunker_semantic.load()
#     #     print(f"Semantic 分割完成，块数: {len(chunks_semantic)}")
#     #     for i, chunk in enumerate(chunks_semantic):
#     #         print(f"--- 块 {i+1} ---")
#     #         print(chunk.page_content)
#     #         print("-" * 10)
#     #
#     # except ImportError as e:
#     #     print(f"Semantic 测试导入失败: {e}")
#     # except Exception as e:
#     #     print(f"Semantic 测试失败: {e}")
