"""
知识库检索器构建组件

负责构建和管理 LangChain 检索器，包括：
- 向量检索器
- BM25 检索器
- 重排序压缩器
- 混合检索器
"""

import logging
from typing import Any, Dict, Literal, Optional

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever

from src.components.reranker_compressor import RemoteRerankerCompressor

logger = logging.getLogger(__name__)


class RetrieverBuilder:
    """
    检索器构建器

    负责构建各种类型的 LangChain 检索器
    """

    # 默认远程重排序模型
    DEFAULT_REMOTE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        use_bm25: bool = False,
        bm25_k: int = 3,
        use_reranker: bool = False,
        reranker_type: Literal["local", "remote"] = "remote",
        remote_rerank_config: Optional[Dict[str, Any]] = None,
        rerank_top_n: int = 3,
    ):
        """
        初始化检索器构建器

        Args:
            use_bm25: 是否启用 BM25 混合检索
            bm25_k: BM25 检索器返回的文档数量
            use_reranker: 是否启用重排序
            reranker_type: 重排序器类型
            remote_rerank_config: 远程重排序配置
            rerank_top_n: 重排序返回的文档数量
        """
        self.use_bm25 = use_bm25
        self.bm25_k = bm25_k
        self.use_reranker = use_reranker
        self.reranker_type = reranker_type
        self.remote_rerank_config = (
            remote_rerank_config if reranker_type == "remote" else None
        )
        self.rerank_top_n = rerank_top_n

        if reranker_type == "remote" and not (
            remote_rerank_config and remote_rerank_config.get("api_key")
        ):
            logger.warning(
                "选择了 'remote' reranker 但未提供有效的 'remote_rerank_config'。"
            )

    def adjust_search_k_for_rerank(self, search_k: int) -> int:
        """
        调整 search_k 参数以适应重排序需求

        Args:
            search_k: 原始 search_k 值

        Returns:
            调整后的 search_k 值
        """
        if self.use_reranker and search_k < self.rerank_top_n:
            logger.warning(
                f"配置的 search_k ({search_k}) 小于 rerank_top_n ({self.rerank_top_n})。"
                f"将增加 search_k 到 {self.rerank_top_n}。"
            )
            return self.rerank_top_n
        return search_k

    def create_vector_retriever(
        self, collection, search_k: int, filter_dict: Optional[dict]
    ) -> BaseRetriever:
        """
        创建基础向量检索器

        Args:
            collection: 向量数据库集合
            search_k: 返回的文档数量
            filter_dict: 元数据过滤字典

        Returns:
            配置好的向量检索器
        """
        search_kwargs = {"k": search_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            logger.info(f"向量检索器应用元数据过滤器: {filter_dict}")

        retriever = collection.as_retriever(search_kwargs=search_kwargs)
        logger.info(f"基础向量检索器配置: {search_kwargs}")
        return retriever

    async def create_bm25_retriever(
        self, kb_id: str, filter_dict: Optional[dict], get_documents_fn
    ) -> Optional[BM25Retriever]:
        """
        创建 BM25 检索器

        Args:
            kb_id: 知识库 ID
            filter_dict: 元数据过滤字典
            get_documents_fn: 获取文档的异步函数

        Returns:
            BM25 检索器实例，如果未启用则返回 None
        """
        if not self.use_bm25:
            return None

        logger.info("BM25 混合检索已启用，正在准备 BM25 检索器...")
        try:
            all_documents = await get_documents_fn(kb_id, filter_dict)

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

    def create_rerank_compressor(self) -> Optional[BaseDocumentCompressor]:
        """
        创建重排序压缩器

        Returns:
            重排序压缩器实例，如果未启用则返回 None
        """
        if not self.use_reranker:
            return None

        logger.info(
            f"启用重排序 (类型: {self.reranker_type}, TopN: {self.rerank_top_n})"
        )

        if self.reranker_type == "remote":
            if self.remote_rerank_config and self.remote_rerank_config.get("api_key"):
                compressor = RemoteRerankerCompressor(
                    api_key=self.remote_rerank_config["api_key"],
                    model_name=self.remote_rerank_config.get(
                        "model", self.DEFAULT_REMOTE_RERANK_MODEL
                    ),
                    top_n=self.rerank_top_n,
                )
                logger.info("远程 RemoteRerankerCompressor 初始化成功。")
                return compressor
            else:
                logger.error("无法初始化远程 Reranker: 缺少 API Key。")

        return None

    def assemble_final_retriever(
        self,
        base_retriever: BaseRetriever,
        bm25_retriever: Optional[BM25Retriever],
        compressor: Optional[BaseDocumentCompressor],
    ) -> BaseRetriever:
        """
        组装最终检索器

        Args:
            base_retriever: 基础向量检索器
            bm25_retriever: BM25 检索器（可选）
            compressor: 重排序压缩器（可选）

        Returns:
            组装好的最终检索器
        """
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
