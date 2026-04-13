"""Retriever tool for knowledge base document retrieval.

This module provides the RAG retrieval tool for agents to search
knowledge bases and retrieve relevant documents.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.components.kb import KnowledgeManager
from src.models.knowledgeBase import KnowledgeBase
from src.utils.embedding import get_embedding
from src.utils.format_doc_list import utils_format_doc_list


class RerankerConfig(BaseModel):
    """重排序器配置(非常建议开启，默认配置即可），用于配置重排序器，包括是否启用重排序、重排序器类型、远程 Reranker 配置和重排序后返回的文档数量"""

    use_reranker: bool = Field(default=True, description="是否启用重排序")
    reranker_type: str = Field(default="remote", description="重排序器类型,默认即可")

    remote_rerank_config: Optional[Dict[str, Any]] = Field(
        default={
            "model": "BAAI/bge-reranker-v2-m3",
            "api_key": "sk-vlscwpfeobatirqzooxdzhwgxeqgdnkvyyirsgumudzbekxb",
        },
        description="远程重排序器配置，默认即可",
    )
    rerank_top_n: int = Field(default=5, ge=1, description="重排序后返回的文档数量")


class KnowledgeConfig(BaseModel):
    knowledge_base_id: str
    filter_by_file_md5: Optional[list[str]] = Field(
        default=None,
        description="文件MD5，用于指定检索的若干个文档，如果为None，则检索所选知识库中的全部文件",
    )
    search_k: Optional[int] = Field(
        default=10, ge=1, description="基础检索器返回的文档数量 (应 >= rerank_top_n)"
    )
    # --- BM25 相关配置 ---
    use_bm25: bool = Field(
        default=True, description="是否启用 BM25 混合检索，默认启用，非常推荐启用"
    )
    bm25_k: int = Field(
        default=3,
        ge=1,
        description="BM25 检索器返回的文档数量,一般3条，值越大，返回的内容越多，参考资料越多，但也可能引入噪声",
    )
    # --- 重排序器配置 ---
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description="重排序器配置，默认启用，非常推荐启用",
    )


class RAGRequest(BaseModel):
    question: str = Field(description="用于检索的原始文本")
    knowledge_config: KnowledgeConfig


@tool
async def retriever_document_tool(request: RAGRequest) -> list[str]:
    """
    知识库检索文档工具，用于检索文档，返回检索到的文档内容,你就可以根据文档内容进行回答.

    Tips:
        1.尝试从知识库列表中找到最有可能包含用户问题的知识库ID（例如：用户问"RAG是什么？"，则最有可能包含答案的知识库是"RAG知识库"），然后传入它对应的id
        2.尝试从知识库文件列表中找到很有可能包含用户问题的若干个文件MD5（例如：用户问"RAG是什么？"，则很可能包含答案的文件是"RAG.pdf和RAG.md"），然后传入此参数，这样缩小的检索范围，提高了检索精度
        3.如果检索不到能够回答问题的文档，可以试试换个问题的问法、扩大检索范围和调整检索参数。
        4.尝试两次检索依旧无果，回复用户"知识库中似乎没有相关内容，需要我为您联网查询吗？"
    """
    knowledge_instance: KnowledgeManager = None
    if request.knowledge_config:
        knowledge_cfg = request.knowledge_config
        try:
            kb = await KnowledgeBase.get(knowledge_cfg.knowledge_base_id)
            if kb.embedding_config:
                _embedding = get_embedding(
                    kb.embedding_config.embedding_supplier,
                    kb.embedding_config.embedding_model,
                    kb.embedding_config.embedding_apikey,
                )
                reranker_cfg = knowledge_cfg.reranker_config
                knowledge_instance = KnowledgeManager(
                    _embeddings=_embedding,
                    splitter="hybrid",
                    # --- BM25 参数 ---
                    use_bm25=knowledge_cfg.use_bm25,
                    bm25_k=knowledge_cfg.bm25_k,
                    # --- Reranker 参数 ---
                    use_reranker=reranker_cfg.use_reranker,
                    reranker_type=reranker_cfg.reranker_type,
                    remote_rerank_config=reranker_cfg.remote_rerank_config,
                    rerank_top_n=reranker_cfg.rerank_top_n,
                )
            else:
                logging.warning(
                    f"未找到知识库 {knowledge_cfg.knowledge_base_id} 或其 embedding 配置。将不初始化 Knowledge 工具。"
                )
        except Exception as e:
            logging.error(
                f"错误：初始化 Knowledge 工具失败 ({e})。将不使用知识库。",
                exc_info=True,
            )
            knowledge_instance = None
    if knowledge_instance:
        _filter_dict = {}
        if knowledge_cfg.filter_by_file_md5 is not None:
            # If filter_by_file_md5 is an empty list, {"$in": []} should correctly match no documents.
            # If it's a non-empty list, it will filter by those MD5s.
            _filter_dict["source_file_md5"] = {"$in": knowledge_cfg.filter_by_file_md5}

        retriever = await knowledge_instance.get_retriever_for_knowledge_base(
            kb_id=knowledge_cfg.knowledge_base_id,
            filter_dict=_filter_dict,
            search_k=knowledge_cfg.search_k,
        )

        context = await retriever.ainvoke(request.question)
        content_json_str = utils_format_doc_list(context)
        logging.info(f"发送给前端的格式化上下文JSON: {content_json_str}")
        return content_json_str
    else:
        return "知识库检索失败，请检查知识库配置是否正确"


__all__ = [
    "retriever_document_tool",
    "RerankerConfig",
    "KnowledgeConfig",
    "RAGRequest",
]
