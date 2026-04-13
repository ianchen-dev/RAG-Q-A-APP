"""
知识库管理器组件

统一的知识库管理接口，整合文档处理和检索器构建功能
"""

import logging
from hashlib import md5
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.components.kb.document_processor import DocumentProcessor
from src.components.kb.retriever_builder import RetrieverBuilder
from src.config.vector_db_config import VectorDBType
from src.factories.vector_db_factory import VectorDBManager

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    知识库管理器

    提供统一的知识库操作接口，包括文档处理和检索器构建
    """

    def __init__(
        self,
        _embeddings=None,
        splitter: str = "hybrid",
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
        # 文档处理配置
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        初始化知识库管理器

        Args:
            _embeddings: 嵌入模型实例
            splitter: 文档分割器类型
            vector_db_type: 向量数据库类型
            use_bm25: 是否启用 BM25 混合检索
            bm25_k: BM25 检索器返回的文档数量
            use_reranker: 是否启用重排序
            reranker_type: 重排序器类型
            remote_rerank_config: 远程重排序配置
            rerank_top_n: 重排序返回的文档数量
            batch_size: 嵌入批次大小，默认64
            max_concurrent_batches: 最大并发批次数量，默认10
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
        """
        self._embeddings = _embeddings
        self.vector_db_type = vector_db_type
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches

        if not self._embeddings:
            logger.warning("KnowledgeManager 在没有提供 embedding 函数的情况下初始化。")

        # 初始化组件
        self.document_processor = DocumentProcessor(
            splitter=splitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever_builder = RetrieverBuilder(
            use_bm25=use_bm25,
            bm25_k=bm25_k,
            use_reranker=use_reranker,
            reranker_type=reranker_type,
            remote_rerank_config=remote_rerank_config,
            rerank_top_n=rerank_top_n,
        )

        # 初始化向量数据库管理器（延迟初始化）
        self._db_manager: Optional[VectorDBManager] = None

        logger.info(
            f"KnowledgeManager 初始化: 向量DB={vector_db_type or '默认'}, "
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
        documents = self.document_processor.load_and_chunk_documents(
            file_path, self._embeddings
        )

        if not documents:
            return

        # 注入元数据
        if is_metadata_to_add:
            processed_documents = self.document_processor.inject_metadata(
                documents, kb_id, file_md5, file_name
            )
        else:
            processed_documents = documents

        # 添加到向量数据库
        collection = self.db_manager.get_collection(kb_id)
        collection_exists = await self.db_manager.collection_exists(kb_id)

        await self.document_processor.add_documents_to_collection(
            collection, kb_id, processed_documents, collection_exists
        )

        # 如果集合不存在，创建它
        if not collection_exists:
            await self.db_manager.create_collection(kb_id)

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
            f"BM25: {'启用' if self.retriever_builder.use_bm25 else '禁用'}, "
            f"Reranker: {'启用' if self.retriever_builder.use_reranker else '禁用'}"
        )

        if not await self.db_manager.collection_exists(kb_id_str):
            error_msg = f"知识库集合 '{kb_id_str}' 不存在！"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 创建基础向量检索器
        collection = self.db_manager.get_collection(kb_id_str)
        effective_search_k = self.retriever_builder.adjust_search_k_for_rerank(search_k)
        base_retriever = self.retriever_builder.create_vector_retriever(
            collection, effective_search_k, filter_dict
        )

        # 创建 BM25 检索器
        bm25_retriever = await self.retriever_builder.create_bm25_retriever(
            kb_id_str, filter_dict, self.db_manager.adapter.get_all_documents
        )

        # 创建重排序压缩器
        try:
            compressor = self.retriever_builder.create_rerank_compressor()
        except Exception as e:
            logger.error(f"创建重排序器时出错: {e}", exc_info=True)
            compressor = None

        # 组装最终检索器
        final_retriever = self.retriever_builder.assemble_final_retriever(
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
