"""
重构后的Knowledge类，采用工厂模式和适配器模式支持多种向量数据库
"""

import logging
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional

from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import (
    BaseDocumentCompressor,
    Document,
)
from langchain_core.retrievers import BaseRetriever

from src.components.reranker_compressor import RemoteRerankerCompressor
from src.config.vector_db_config import VectorDBType
from src.factories.vector_db_factory import VectorDBManager
from src.utils.DocumentChunker import DocumentChunker

# 配置日志
logger = logging.getLogger(__name__)


# --- 重构后的 Knowledge 类 ---
class Knowledge:
    """
    重构后的知识库工具类，使用工厂模式和适配器模式支持多种向量数据库
    """

    def __init__(
        self,
        _embeddings=None,
        splitter="hybrid",
        # 向量数据库配置
        vector_db_type: Optional[VectorDBType] = None,
        # BM25 相关配置
        use_bm25: bool = False,
        bm25_k: int = 3,
        # 重排序相关配置
        use_reranker: bool = False,
        reranker_type: Literal["local", "remote"] = "remote",
        remote_rerank_config: Optional[Dict[str, Any]] = None,
        rerank_top_n: int = 3,
        # 批次处理配置
        batch_size: int = 64,
        max_concurrent_batches: int = 10,
    ):
        """
        初始化Knowledge类

        Args:
            _embeddings: 嵌入模型实例
            splitter: 文档分割器类型
            vector_db_type: 向量数据库类型
            use_bm25: 是否启用BM25混合检索
            bm25_k: BM25检索器返回的文档数量
            use_reranker: 是否启用重排序
            reranker_type: 重排序器类型
            remote_rerank_config: 远程重排序配置
            rerank_top_n: 重排序返回的文档数量
            batch_size: 嵌入批次大小，默认64
            max_concurrent_batches: 最大并发批次数量，默认5
        """
        self._embeddings = _embeddings
        self.splitter = splitter
        self.vector_db_type = vector_db_type
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

        if not self._embeddings:
            logger.warning("Knowledge 类在没有提供 embedding 函数的情况下初始化。")

        # 存储配置
        self.use_bm25 = use_bm25
        self.bm25_k = bm25_k
        self.use_reranker = use_reranker
        self.reranker_type = reranker_type
        self.remote_rerank_config = (
            remote_rerank_config if reranker_type == "remote" else None
        )
        if reranker_type == "remote" and not (
            remote_rerank_config and remote_rerank_config.get("api_key")
        ):
            logger.warning(
                "选择了 'remote' reranker 但未提供有效的 'remote_rerank_config'。"
            )
        self.rerank_top_n = rerank_top_n

        # 初始化向量数据库管理器（延迟初始化）
        self._db_manager: Optional[VectorDBManager] = None

        logger.info(
            f"Knowledge 初始化: 向量DB={vector_db_type or '默认'}, "
            f"BM25={'启用' if use_bm25 else '禁用'}, "
            f"Reranker={'启用' if use_reranker else '禁用'}, "
            f"批次大小={batch_size}, 最大并发批次={max_concurrent_batches}"
        )

    @property
    def db_manager(self) -> VectorDBManager:
        """获取向量数据库管理器实例（延迟初始化）"""
        if self._db_manager is None:
            if not self._embeddings:
                raise ValueError(
                    "无法初始化向量数据库管理器，因为缺少 embedding 函数。"
                )
            self._db_manager = VectorDBManager(
                self._embeddings,
                self.vector_db_type,
                batch_size=self.batch_size,
                max_concurrent_batches=self.max_concurrent_batches,
            )
        return self._db_manager

    async def collection_exists(self, collection_name: str) -> bool:
        """检查指定集合名称的向量数据库是否存在"""
        return await self.db_manager.collection_exists(collection_name)

    def load_knowledge(self, collection_name: str):
        """加载指定名称的向量存储"""
        if not self._embeddings:
            raise ValueError("无法加载知识库，因为缺少 embedding 函数。")

        logger.info(f"加载集合 '{collection_name}'")
        return self.db_manager.get_collection(collection_name)

    async def add_file_to_knowledge_base(
        self,
        kb_id: str,
        file_path: str,
        file_name: str,
        file_md5: str,
        is_metadata_to_add: bool = False,
    ) -> None:
        """
        异步将单个文件处理并添加到指定的知识库集合中

        Args:
            kb_id: 知识库ID，将作为集合名称
            file_path: 要处理的文件路径
            file_name: 原始文件名
            file_md5: 文件的MD5值，用于元数据
            is_metadata_to_add: 是否添加元数据到文档块，默认False
        """
        logger.info(
            f"开始处理文件 {file_path} (MD5: {file_md5}) 并添加到知识库 {kb_id}..."
        )
        if not self._embeddings:
            raise ValueError("无法处理文件，因为缺少 embedding 函数。")

        # 加载和分块文档
        try:
            logger.debug(
                f"使用 DocumentChunker (类型: {self.splitter}) 加载和分块: {file_path}"
            )
            loader = DocumentChunker(
                file_path,
                splitter_type=self.splitter,
                embeddings=self._embeddings,
                chunk_size=500,  # 减小chunk_size以避免token超限问题
                chunk_overlap=50,  # 相应减小overlap
            )

            documents = loader.load()
            if not documents:
                logger.warning(f"警告: 文件 {file_path} 未产生任何文档块，跳过处理。")
                return
            logger.info(f"文件 {file_path} 加载并分块完成，共 {len(documents)} 块。")
        except Exception as e:
            logger.error(f"加载或分块文件 {file_path} 时出错: {e}", exc_info=True)
            raise
        logger.debug(f"是否添加元数据: {is_metadata_to_add}")
        processed_documents = []
        if is_metadata_to_add:
            # 准备并注入元数据
            metadata_to_add = {
                "knowledge_base_id": str(kb_id),
                # "source_file_path": file_path,
                "source_file_md5": file_md5,
                "source_file_name": file_name,
            }
            logger.debug(f"为文档块添加元数据: {metadata_to_add}")

            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                current_metadata = doc.metadata.copy()
                current_metadata.update(metadata_to_add)
                processed_documents.append(
                    Document(page_content=doc.page_content, metadata=current_metadata)
                )
        else:
            processed_documents = documents
        # 添加到向量数据库
        try:
            collection = self.db_manager.get_collection(kb_id)

            if not await self.db_manager.collection_exists(kb_id):
                logger.info(f"集合 '{kb_id}' 不存在，首次创建并添加文档...")
                await self.db_manager.create_collection(kb_id)
            else:
                logger.info(f"集合 '{kb_id}' 已存在，加载并添加新文档...")

            await collection.aadd_documents(processed_documents)
            logger.info(f"文件 {file_path} 的向量数据成功添加到集合 '{kb_id}'。")

        except Exception as e:
            logger.error(
                f"将文件 {file_path} 的向量数据添加到集合 '{kb_id}' 时出错: {e}",
                exc_info=True,
            )
            raise

    def _adjust_search_k_for_rerank(self, search_k: int) -> int:
        """调整 search_k 参数以适应重排序需求"""
        if self.use_reranker and search_k < self.rerank_top_n:
            logger.warning(
                f"配置的 search_k ({search_k}) 小于 rerank_top_n ({self.rerank_top_n})。"
                f"将增加 search_k 到 {self.rerank_top_n}。"
            )
            return self.rerank_top_n
        return search_k

    def _create_vector_retriever(
        self, collection, search_k: int, filter_dict: Optional[dict]
    ) -> BaseRetriever:
        """创建基础向量检索器"""
        search_kwargs = {"k": search_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            logger.info(f"向量检索器应用元数据过滤器: {filter_dict}")

        retriever = collection.as_retriever(search_kwargs=search_kwargs)
        logger.info(f"基础向量检索器配置: {search_kwargs}")
        return retriever

    async def _create_bm25_retriever(
        self, kb_id: str, filter_dict: Optional[dict]
    ) -> Optional[BM25Retriever]:
        """创建 BM25 检索器"""
        if not self.use_bm25:
            return None

        logger.info("BM25 混合检索已启用，正在准备 BM25 检索器...")
        try:
            all_documents = await self.db_manager.adapter.get_all_documents(
                kb_id, filter_dict
            )

            if not all_documents:
                logger.warning(
                    f"从集合 '{kb_id}' 未获取到任何文档，无法创建 BM25 检索器。"
                )
                return None

            retriever = BM25Retriever.from_documents(all_documents)
            retriever.k = self.bm25_k
            logger.info(f"BM25 检索器初始化成功，k={self.bm25_k}。")
            return retriever

        except Exception as e:
            logger.error(f"初始化 BM25 检索器时出错: {e}", exc_info=True)
            return None

    def _create_rerank_compressor(self) -> Optional[BaseDocumentCompressor]:
        """创建重排序压缩器"""
        if not self.use_reranker:
            return None

        logger.info(
            f"启用重排序 (类型: {self.reranker_type}, TopN: {self.rerank_top_n})"
        )

        if self.reranker_type == "remote":
            if (
                self.remote_rerank_config
                and self.remote_rerank_config.get("api_key")
            ):
                compressor = RemoteRerankerCompressor(
                    api_key=self.remote_rerank_config["api_key"],
                    model_name=self.remote_rerank_config.get(
                        "model", RemoteRerankerCompressor.model_name
                    ),
                    top_n=self.rerank_top_n,
                )
                logger.info("远程 RemoteRerankerCompressor 初始化成功。")
                return compressor
            else:
                logger.error("无法初始化远程 Reranker: 缺少 API Key。")

        return None

    def _assemble_final_retriever(
        self,
        base_retriever: BaseRetriever,
        bm25_retriever: Optional[BM25Retriever],
        compressor: Optional[BaseDocumentCompressor],
    ) -> BaseRetriever:
        """组装最终检索器"""
        # 应用重排序
        if compressor:
            vector_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )
        else:
            vector_retriever = base_retriever

        # 应用 BM25 混合检索
        if bm25_retriever:
            return EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
            )

        return vector_retriever

    async def get_retriever_for_knowledge_base(
        self, kb_id: str, filter_dict: Optional[dict] = None, search_k: int = 3
    ) -> BaseRetriever:
        """
        异步根据知识库ID获取检索器
        支持可选的元数据过滤、BM25混合检索和重排序

        Args:
            kb_id: 知识库ID，即集合名称
            filter_dict: 用于元数据过滤的字典
            search_k: 向量检索器返回的文档数量

        Returns:
            配置好的 Langchain BaseRetriever
        """
        kb_id_str = str(kb_id)
        logger.info(
            f"开始为知识库 '{kb_id_str}' 获取检索器... "
            f"BM25: {'启用' if self.use_bm25 else '禁用'}, "
            f"Reranker: {'启用' if self.use_reranker else '禁用'}"
        )

        if not await self.db_manager.collection_exists(kb_id_str):
            error_msg = f"知识库集合 '{kb_id_str}' 不存在！"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 创建基础向量检索器
        collection = self.db_manager.get_collection(kb_id_str)
        effective_search_k = self._adjust_search_k_for_rerank(search_k)
        base_retriever = self._create_vector_retriever(
            collection, effective_search_k, filter_dict
        )

        # 创建 BM25 检索器
        bm25_retriever = await self._create_bm25_retriever(kb_id_str, filter_dict)

        # 创建重排序压缩器
        try:
            compressor = self._create_rerank_compressor()
        except Exception as e:
            logger.error(f"创建重排序器时出错: {e}", exc_info=True)
            compressor = None

        # 组装最终检索器
        final_retriever = self._assemble_final_retriever(
            base_retriever, bm25_retriever, compressor
        )

        logger.info(f"最终返回的检索器类型: {type(final_retriever)}")
        return final_retriever

    async def retrieve_documents(self, kb_id: str, query: str) -> List[Document]:
        """异步检索文档"""
        retriever = await self.get_retriever_for_knowledge_base(kb_id)
        return await retriever.ainvoke(query)

    async def delete_collection(self, kb_id: str) -> bool:
        """删除指定的知识库集合"""
        return await self.db_manager.delete_collection(kb_id)

    async def delete_documents_by_filter(
        self, kb_id: str, filter_dict: Dict[str, Any]
    ) -> bool:
        """根据过滤条件删除文档"""
        return await self.db_manager.adapter.delete_documents(kb_id, filter_dict)

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

    async def close(self):
        """关闭连接，清理资源"""
        if self._db_manager:
            await self._db_manager.close()
            self._db_manager = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()
