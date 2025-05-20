import logging  # 添加日志记录
import os
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional, Sequence  # 更新 typing

from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_chroma import Chroma

# from langchain_community.cross_encoders import HuggingFaceCrossEncoder #discard
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import Callbacks  # Callbacks for compressor
from langchain_core.documents import (
    BaseDocumentCompressor,  # 导入基类
    Document,
)
from langchain_core.retrievers import BaseRetriever

from src.utils.DocumentChunker import DocumentChunker
from src.utils.remote_rerank import call_siliconflow_rerank

# 配置日志
logger = logging.getLogger(__name__)

# 设置知识库 向量模型 重排序模型的路径
# DEFAULT_LOCAL_RERANK_MODEL = "src/utils/bge-reranker-large"  # 本地重排序模型路径--discard
DEFAULT_REMOTE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # 默认远程模型
chroma_dir = "chroma/"  # 向量数据库的路径

# --- 自定义远程 Reranker Compressor ---


class RemoteRerankerCompressor(BaseDocumentCompressor):
    """
    一个自定义的 Langchain 文档压缩器，
    通过调用 SiliconFlow API 来对文档进行重排序。

    注意：此类继承自 Langchain 的 BaseDocumentCompressor，
    其配置参数通过类属性定义，由 Pydantic 处理初始化。
    """

    # 将参数定义为类属性，而不是在 __init__ 中
    api_key: str
    "SiliconFlow API 密钥。"  # Docstring for attribute
    model_name: str = DEFAULT_REMOTE_RERANK_MODEL
    "要使用的 SiliconFlow Rerank 模型名称。"  # Docstring for attribute
    top_n: int = 3
    "返回最相关的 top_n 个文档。"  # Docstring for attribute

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        异步压缩文档，通过调用 SiliconFlow API 进行重排序。

        Args:
            documents: 从基础检索器获取的文档序列。
            query: 用户的原始查询。
            callbacks: Langchain 回调。

        Returns:
            根据 SiliconFlow API 重排序后的文档序列。
        """
        if not documents:
            return []
        if not self.api_key:
            logger.error("缺少 SiliconFlow API key，无法执行远程重排序。")
            # 在这种情况下，可以选择返回原始文档或空列表
            # 返回原始文档可能更好，避免完全丢失信息
            return documents

        doc_contents = [doc.page_content for doc in documents]
        logger.debug(
            f"调用 SiliconFlow Rerank: query='{query[:50]}...', docs_count={len(doc_contents)}"
        )

        # 调用我们之前定义的 remote_rerank 函数
        ranked_results = await call_siliconflow_rerank(
            api_key=self.api_key,
            query=query,
            documents=doc_contents,
            model=self.model_name,
            top_n=self.top_n,  # 传递 top_n
        )

        final_docs = []
        if ranked_results:
            logger.debug(f"SiliconFlow Rerank 返回 {len(ranked_results)} 个结果。")
            for result in ranked_results:
                original_index = result.get("index")
                score = result.get("relevance_score")
                if original_index is not None and 0 <= original_index < len(documents):
                    # 获取原始文档并添加分数到 metadata
                    original_doc = documents[original_index]
                    # 创建 metadata 副本以避免修改原始对象
                    new_metadata = (
                        original_doc.metadata.copy() if original_doc.metadata else {}
                    )
                    new_metadata["relevance_score"] = score  # 添加相关性分数
                    # 创建新的 Document 对象，包含更新后的 metadata
                    final_docs.append(
                        Document(
                            page_content=original_doc.page_content,
                            metadata=new_metadata,
                        )
                    )
                else:
                    logger.warning(f"SiliconFlow 返回了无效的索引: {original_index}")
            logger.info(f"远程重排序完成，返回 {len(final_docs)} 个文档。")
        else:
            logger.warning("远程 Rerank 调用失败或未返回有效结果，将返回原始文档。")
            # 在失败时返回原始文档的前 top_n 个，或全部返回？
            # 暂时返回所有原始文档以避免信息丢失
            return documents

        return final_docs

    # 同步版本可以简单地引发 NotImplementedError 或尝试包装异步版本
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """同步版本，当前未完全实现，依赖于异步版本。"""
        # 这是一个简化的实现，可能在某些环境中不起作用
        # 推荐主要使用异步流程
        try:
            import asyncio

            # 尝试获取或创建事件循环
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # 在事件循环中运行异步方法
            result = loop.run_until_complete(
                self.acompress_documents(documents, query, callbacks)
            )
            # 如果创建了新循环，需要关闭它
            # if not asyncio.get_event_loop().is_running(): # Python 3.7+
            #    loop.close()
            return result
        except ImportError:
            raise NotImplementedError(
                "需要 asyncio 库来运行同步版本的 acompress_documents。"
            )
        except Exception as e:
            logger.error(f"运行同步 compress_documents 时出错: {e}")
            # 返回原始文档作为后备
            return documents


# --- 更新 Knowledge 类 ---


class Knowledge:
    """知识库工具类，处理向量化、存储和检索"""

    def __init__(
        self,
        _embeddings=None,
        splitter="hybrid",
        # splitter="semantic",
        # --- BM25 相关配置 ---
        use_bm25: bool = False,  # 是否启用 BM25 混合检索
        bm25_k: int = 3,  # BM25 检索器返回的文档数量
        # --- 重排序相关配置 ---
        use_reranker: bool = False,  # 是否启用重排序，替代旧的 reorder
        reranker_type: Literal["local", "remote"] = "remote",  # 重排序器类型
        # local_rerank_model_path: str = DEFAULT_LOCAL_RERANK_MODEL,  # 本地模型路径--discard
        remote_rerank_config: Optional[Dict[str, Any]] = None,  # 远程配置字典
        rerank_top_n: int = 3,  # 返回的文档数量
    ):
        self._embeddings = _embeddings
        self.splitter = splitter
        if not self._embeddings:
            logger.warning("Knowledge 类在没有提供 embedding 函数的情况下初始化。")

        # --- 存储 BM25 配置 ---
        self.use_bm25 = use_bm25
        self.bm25_k = bm25_k

        # --- 存储重排序配置 ---
        self.use_reranker = use_reranker
        self.reranker_type = reranker_type
        # self.local_rerank_model_path = local_rerank_model_path
        # 验证远程配置
        self.remote_rerank_config = (
            remote_rerank_config if reranker_type == "remote" else None
        )
        if reranker_type == "remote" and not (
            remote_rerank_config and remote_rerank_config.get("api_key")
        ):
            logger.warning(
                "选择了 'remote' reranker 但未提供有效的 'remote_rerank_config' (包含 'api_key')。重排序可能无法工作。"
            )
            # 可以考虑禁用重排序 self.use_reranker = False
        self.rerank_top_n = rerank_top_n

        logger.info(
            f"Knowledge 初始化: BM25={'启用' if use_bm25 else '禁用'}, BM25_k={bm25_k if use_bm25 else 'N/A'}, "
            f"Reranker={'启用' if use_reranker else '禁用'}, Type={reranker_type if use_reranker else 'N/A'}, TopN={rerank_top_n if use_reranker else 'N/A'}"
        )

    @staticmethod
    def is_already_vector_database(collection_name: str) -> bool:
        """检查指定集合名称的 ChromaDB 物理存储是否存在"""
        persist_directory = os.path.join(chroma_dir, collection_name)
        return os.path.isdir(persist_directory)

    def load_knowledge(self, collection_name) -> Chroma:
        """加载指定名称的 Chroma 向量数据库"""
        if not self._embeddings:
            raise ValueError("无法加载知识库，因为缺少 embedding 函数。")
        persist_directory = os.path.join(chroma_dir, collection_name)
        logger.info(f"尝试从 '{persist_directory}' 加载集合 '{collection_name}'")
        return Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self._embeddings,
        )

    async def add_file_to_knowledge_base(
        self, kb_id: str, file_path: str, file_name: str, file_md5: str
    ) -> None:
        """
        异步将单个文件处理并添加到指定的知识库集合中（集合名为 kb_id）。
        :param kb_id: 知识库ID，将作为 Chroma 的 collection_name。
        :param file_path: 要处理的文件路径。
        :param file_name: 原始文件名。
        :param file_md5: 文件的MD5值，用于元数据。
        """
        logger.info(
            f"开始处理文件 {file_path} (MD5: {file_md5}) 并添加到知识库 {kb_id}..."
        )
        if not self._embeddings:
            raise ValueError("无法处理文件，因为缺少 embedding 函数。")

        # --- 1. 加载和分块文档 ---
        try:
            logger.debug(
                f"使用 DocumentChunker (类型: {self.splitter}) 加载和分块: {file_path}"
            )
            loader = DocumentChunker(
                file_path,
                splitter_type=self.splitter,  # 'hybrid' 或其他选项
                embeddings=self._embeddings,  # 对于 'hybrid' 和 'semantic' 模式需要
                chunk_size=500,  # 可以根据需要调整
                chunk_overlap=50,  # 可以根据需要调整
            )

            documents = loader.load()
            if not documents:
                logger.warning(f"警告: 文件 {file_path} 未产生任何文档块，跳过处理。")
                return
            logger.info(f"文件 {file_path} 加载并分块完成，共 {len(documents)} 块。")
        except ImportError as e:
            logger.error(
                f"错误：看起来缺少使用 SemanticChunker 所需的库: {e}", exc_info=True
            )
            logger.error(
                "请尝试运行: pdm add langchain_experimental sentence-transformers bert_score"
            )
            raise
        except ValueError as e:
            # 捕获 DocumentChunker 内部抛出的 ValueError，例如 embeddings 未提供
            logger.error(f"处理文档时发生配置错误: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"加载或分块文件 {file_path} 时出错: {e}", exc_info=True)
            raise

        # --- 2. 准备并注入元数据 ---
        metadata_to_add = {
            "knowledge_base_id": str(kb_id),  # 确保是字符串
            "source_file_path": file_path,
            "source_file_md5": file_md5,
            "source_file_name": file_name,
        }
        logger.debug(f"为文档块添加元数据: {metadata_to_add}")
        processed_documents = []
        for doc in documents:
            if doc.metadata is None:
                doc.metadata = {}
            # 更新元数据，使用 .copy() 避免意外修改原始 metadata_to_add
            current_metadata = doc.metadata.copy()
            current_metadata.update(metadata_to_add)
            # 创建一个新的 Document 或直接修改，取决于 DocumentChunker 实现
            # 为安全起见，可以创建新 Document
            processed_documents.append(
                Document(page_content=doc.page_content, metadata=current_metadata)
            )
            # 或者如果可以直接修改: doc.metadata.update(metadata_to_add)

        # --- 3. 添加到 ChromaDB ---
        kb_id_str = str(kb_id)  # 确保是字符串
        persist_directory = os.path.join(chroma_dir, kb_id_str)

        try:
            if not self.is_already_vector_database(kb_id_str):
                logger.info(f"集合 '{kb_id_str}' 不存在，首次创建并添加文档...")
                # 首次创建
                await Chroma.afrom_documents(
                    documents=processed_documents,  # 使用处理过的文档
                    embedding=self._embeddings,
                    collection_name=kb_id_str,
                    persist_directory=persist_directory,
                )
                logger.info(f"集合 '{kb_id_str}' 创建成功。")
            else:
                logger.info(f"集合 '{kb_id_str}' 已存在，加载并添加新文档...")
                # 集合已存在，加载后添加
                vectorstore = self.load_knowledge(kb_id_str)
                await vectorstore.aadd_documents(
                    documents=processed_documents
                )  # 使用处理过的文档
                logger.info(f"新文档块已添加到现有集合 '{kb_id_str}'。")

            logger.info(
                f"文件 {file_path} 的向量数据成功添加/更新到集合 '{kb_id_str}'。"
            )

        except Exception as e:
            logger.error(
                f"将文件 {file_path} 的向量数据添加到集合 '{kb_id_str}' 时出错: {e}",
                exc_info=True,
            )
            raise

    async def get_retriever_for_knowledge_base(
        self, kb_id: str, filter_dict: Optional[dict] = None, search_k: int = 3
    ) -> BaseRetriever:
        """
        异步根据知识库ID (kb_id) 获取检索器。
        支持可选的元数据过滤、BM25混合检索和重排序。

        :param kb_id: 知识库ID，即 Chroma 集合名称。
        :param filter_dict: 用于元数据过滤的字典 (例如 {"source_file_md5": "..."})。
        :param search_k: 向量检索器返回的文档数量。
        :return: 配置好的 Langchain BaseRetriever。
        """
        kb_id_str = str(kb_id)
        logger.info(
            f"开始为知识库 '{kb_id_str}' 获取检索器... "
            f"BM25: {'启用' if self.use_bm25 else '禁用'} (k={self.bm25_k}), "
            f"Reranker: {'启用' if self.use_reranker else '禁用'} (Type: {self.reranker_type}, TopN: {self.rerank_top_n})"
        )

        if not self.is_already_vector_database(kb_id_str):
            error_msg = f"知识库集合 '{kb_id_str}' 的物理存储 (在 {chroma_dir}) 不存在或无法访问！"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # --- 1. 加载 Chroma Vector Store ---
        logger.info(f"加载知识库 '{kb_id_str}'...")
        try:
            vectorstore = self.load_knowledge(kb_id_str)
        except Exception as e:
            error_msg = f"加载知识库 '{kb_id_str}' 时出错: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # --- 2. 初始化基础向量检索器 (base_retriever) ---
        effective_search_k = search_k
        if self.use_reranker and search_k < self.rerank_top_n:
            logger.warning(
                f"配置的 search_k ({search_k}) 小于 rerank_top_n ({self.rerank_top_n})。将增加 search_k 到 {self.rerank_top_n}。"
                f"建议设置更大的 search_k (如 {self.rerank_top_n * 2}-{self.rerank_top_n * 5}) 以获得更好的重排效果。"
            )
            effective_search_k = self.rerank_top_n
        elif self.use_reranker and search_k < self.rerank_top_n * 2:
            logger.info(
                f"search_k ({search_k}) 略小于 rerank_top_n ({self.rerank_top_n}) 的推荐倍数。考虑增加 search_k 以获得更好的重排效果。"
            )

        vector_search_kwargs = {"k": effective_search_k}
        if filter_dict:
            vector_search_kwargs["filter"] = filter_dict
            logger.info(f"向量检索器应用元数据过滤器: {filter_dict}")
        else:
            logger.info("向量检索器不应用元数据过滤器")

        base_retriever = vectorstore.as_retriever(search_kwargs=vector_search_kwargs)
        logger.info(f"基础向量检索器 (Chroma) 配置: {vector_search_kwargs}")

        # --- 3. 初始化 BM25 检索器 (如果启用) ---
        bm25_retriever: Optional[BM25Retriever] = None
        if self.use_bm25:
            logger.info("BM25 混合检索已启用，正在准备 BM25 检索器...")
            try:
                # 使用 aget 获取文档内容和元数据
                # 注意: where 参数用于元数据过滤，我们传入 filter_dict
                logger.info(
                    f"使用 aget 从 Chroma 集合 '{kb_id_str}' 获取文档以初始化 BM25..."
                )
                logger.info(
                    f"aget 的 where 过滤器: {filter_dict}"
                )  # filter_dict 可能为 None
                chroma_get_result = (
                    vectorstore.get(  # 只有get同步方法——考虑后续使用mongodb
                        where=filter_dict,  # 直接使用 filter_dict 进行元数据过滤
                        include=["documents", "metadatas"],  # 只获取需要的内容
                    )
                )

                # 检查返回结果
                if not chroma_get_result or not chroma_get_result.get("documents"):
                    logger.warning(
                        f"从 Chroma 集合 '{kb_id_str}' 未获取到任何文档 (可能集合为空或过滤条件严格)。无法创建 BM25 检索器。"
                    )
                else:
                    # 将 Chroma 返回结果转换为 Langchain Document 列表
                    all_docs_from_chroma: List[Document] = []
                    doc_contents = chroma_get_result.get("documents", [])
                    metadatas = chroma_get_result.get("metadatas", [])
                    # ids = chroma_get_result.get("ids", [])  # 获取 ids 以备将来可能使用

                    # 确保内容和元数据列表长度一致 (理论上 Chroma 会保证)
                    if len(doc_contents) == len(metadatas):
                        for content, meta in zip(doc_contents, metadatas):
                            # 创建 Document 对象，meta 可能为 None
                            all_docs_from_chroma.append(
                                Document(
                                    page_content=content, metadata=meta if meta else {}
                                )
                            )
                        logger.info(
                            f"成功从 Chroma 获取 {len(all_docs_from_chroma)} 个文档用于 BM25。"
                        )

                        # 初始化 BM25Retriever
                        bm25_retriever = BM25Retriever.from_documents(
                            all_docs_from_chroma
                        )
                        bm25_retriever.k = self.bm25_k  # 设置 BM25 返回数量
                        logger.info(f"BM25 检索器初始化成功，k={self.bm25_k}。")
                    else:
                        logger.error(
                            f"从 Chroma get 返回的 documents ({len(doc_contents)}) 和 metadatas ({len(metadatas)}) 数量不匹配！无法创建 BM25 检索器。"
                        )

            except Exception as e:
                logger.error(
                    f"从 Chroma 获取文档或初始化 BM25 检索器时出错: {e}", exc_info=True
                )
                # 出错时，bm25_retriever 保持为 None，后续逻辑将只使用向量检索

        # --- 4. 确定最终检索器 ---
        final_retriever: BaseRetriever

        if self.use_reranker:
            # --- 场景 A: 使用重排序 ---
            logger.info(
                f"启用重排序 (类型: {self.reranker_type}, TopN: {self.rerank_top_n})，正在配置 ContextualCompressionRetriever..."
            )
            compressor: Optional[BaseDocumentCompressor] = None
            # ... (现有创建 compressor 的逻辑不变，基于 self.reranker_type) ...
            try:
                # 本地重排序模型--discard
                # if self.reranker_type == "local":
                #     # ... (加载本地 reranker) ...
                #     encoder_model = HuggingFaceCrossEncoder(
                #         model_name=self.local_rerank_model_path,
                #         model_kwargs={"device": "cpu"},
                #     )  # 示例
                #     compressor = CrossEncoderReranker(
                #         model=encoder_model, top_n=self.rerank_top_n
                #     )
                #     logger.info("本地 CrossEncoderReranker 初始化成功。")
                if self.reranker_type == "remote":
                    # ... (加载远程 reranker) ...
                    if self.remote_rerank_config and self.remote_rerank_config.get(
                        "api_key"
                    ):
                        compressor = RemoteRerankerCompressor(
                            api_key=self.remote_rerank_config["api_key"],
                            model_name=self.remote_rerank_config.get(
                                "model", DEFAULT_REMOTE_RERANK_MODEL
                            ),
                            top_n=self.rerank_top_n,
                        )
                        logger.info("远程 RemoteRerankerCompressor 初始化成功。")
                    else:
                        logger.error("无法初始化远程 Reranker: 缺少 API Key。")
                        # 这里 compressor 会是 None

                # 如果 compressor 创建失败，打印警告
                if not compressor:
                    logger.warning("未能创建 Reranker Compressor。将跳过重排序步骤。")
                    # 如果 reranker 创建失败，退回到无 reranker 的逻辑
                    # 检查是否需要 BM25
                    if self.use_bm25 and bm25_retriever:
                        logger.info(
                            "Reranker失败，但启用BM25。使用 EnsembleRetriever 组合 BM25 和基础向量检索器。"
                        )
                        final_retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, base_retriever],
                            weights=[0.5, 0.5],
                        )
                    else:
                        logger.info(
                            "Reranker失败，且未启用BM25。仅使用基础向量检索器。"
                        )
                        final_retriever = base_retriever

                else:
                    # Compressor 创建成功，创建 ContextualCompressionRetriever
                    reranked_vector_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=base_retriever
                    )
                    logger.info(
                        "ContextualCompressionRetriever (重排序向量检索器) 创建成功。"
                    )

                    # 根据是否启用 BM25 组合
                    if self.use_bm25 and bm25_retriever:
                        logger.info(
                            "启用重排序和BM25。使用 EnsembleRetriever 组合 BM25 和重排序后的向量检索器。"
                        )
                        # 组合 BM25 和 重排序后的向量检索器
                        final_retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, reranked_vector_retriever],
                            weights=[
                                0.5,
                                0.5,
                            ],  # weights 可以调整或移除，RRF(d) = Σ 1/(k + r_i)
                        )
                    else:
                        logger.info(
                            "启用重排序，但未启用BM25 (或BM25初始化失败)。仅使用重排序后的向量检索器。"
                        )
                        final_retriever = reranked_vector_retriever

            except Exception as e:
                logger.error(
                    f"创建重排序 Compressor 或 ContextualCompressionRetriever 时出错: {e}",
                    exc_info=True,
                )
                # 出错时，回退到基础检索器或 BM25+基础检索器
                logger.warning("重排序流程出错，将回退。")
                if self.use_bm25 and bm25_retriever:
                    logger.info(
                        "回退：使用 EnsembleRetriever 组合 BM25 和基础向量检索器。"
                    )
                    final_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, base_retriever], weights=[0.5, 0.5]
                    )
                else:
                    logger.info("回退：仅使用基础向量检索器。")
                    final_retriever = base_retriever

        else:
            # --- 场景 B: 不使用重排序 ---
            logger.info("重排序未启用。")
            if self.use_bm25 and bm25_retriever:
                logger.info(
                    "未启用重排序，但启用BM25。使用 EnsembleRetriever 组合 BM25 和基础向量检索器。"
                )
                # 组合 BM25 和 基础向量检索器
                final_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, base_retriever],
                    weights=[0.5, 0.5],  # weights 可以调整或移除
                )
            else:
                logger.info(
                    "未启用重排序，且未启用BM25 (或BM25初始化失败)。仅使用基础向量检索器。"
                )
                final_retriever = base_retriever

        logger.info(f"最终返回的检索器类型: {type(final_retriever)}")
        return final_retriever

    @staticmethod
    def get_file_md5(file_path: str) -> str:
        """对文件内容计算md5值"""
        logger.debug(f"计算文件 MD5: {file_path}")
        block_size = 65536
        m = md5()
        try:
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(block_size)
                    if not data:
                        break
                    m.update(data)
            hex_digest = m.hexdigest()
            logger.debug(f"文件 MD5 计算完成 ({file_path}): {hex_digest}")
            return hex_digest
        except FileNotFoundError:
            logger.error(f"错误: 文件未找到 {file_path}")
            raise FileNotFoundError(f"无法计算 MD5，文件未找到: {file_path}")
        except Exception as e:
            logger.error(f"计算文件 {file_path} MD5 时发生未知错误: {e}", exc_info=True)
            raise RuntimeError(f"计算文件 {file_path} MD5 时出错: {e}") from e
