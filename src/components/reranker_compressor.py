"""
Remote Reranker Compressor component for document reranking.

This module provides a custom LangChain document compressor that uses
SiliconFlow API to rerank documents based on query relevance.
"""

import logging
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from src.utils.remote_rerank import call_siliconflow_rerank

# Configure logger
logger = logging.getLogger(__name__)

# Default rerank model
DEFAULT_REMOTE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class RemoteRerankerCompressor(BaseDocumentCompressor):
    """
    A custom LangChain document compressor that reranks documents
    by calling the SiliconFlow API.

    This compressor is used in RAG pipelines to improve retrieval quality
    by reranking retrieved documents based on their relevance to the query.

    Attributes:
        api_key: The API key for SiliconFlow service
        model_name: The model to use for reranking (default: BAAI/bge-reranker-v2-m3)
        top_n: Number of top documents to return after reranking
    """

    api_key: str
    model_name: str = DEFAULT_REMOTE_RERANK_MODEL
    top_n: int = 3

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Asynchronously compress documents by calling SiliconFlow API for reranking.

        Args:
            documents: Sequence of LangChain Document objects to rerank
            query: The query string to rank documents against
            callbacks: Optional callbacks for the operation

        Returns:
            Sequence of reranked Document objects with relevance scores in metadata
        """
        if not documents:
            return []
        if not self.api_key:
            logger.error(
                "Missing SiliconFlow API key, cannot perform remote reranking."
            )
            return documents

        doc_contents = [doc.page_content for doc in documents]
        logger.debug(
            f"Calling SiliconFlow Rerank: query='{query[:50]}...', docs_count={len(doc_contents)}"
        )

        # Call remote reranking function
        ranked_results = await call_siliconflow_rerank(
            api_key=self.api_key,
            query=query,
            documents=doc_contents,
            model=self.model_name,
            top_n=self.top_n,
        )

        final_docs = []
        if ranked_results:
            logger.debug(f"SiliconFlow Rerank returned {len(ranked_results)} results.")
            for result in ranked_results:
                original_index = result.get("index")
                score = result.get("relevance_score")
                if original_index is not None and 0 <= original_index < len(documents):
                    original_doc = documents[original_index]
                    new_metadata = (
                        original_doc.metadata.copy() if original_doc.metadata else {}
                    )
                    new_metadata["relevance_score"] = score
                    final_docs.append(
                        Document(
                            page_content=original_doc.page_content,
                            metadata=new_metadata,
                        )
                    )
                else:
                    logger.warning(
                        f"SiliconFlow returned invalid index: {original_index}"
                    )
            logger.info(
                f"Remote reranking completed, returning {len(final_docs)} documents."
            )
        else:
            logger.warning(
                "Remote Rerank call failed or returned no valid results, returning original documents."
            )
            return documents

        return final_docs

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Synchronous wrapper for the asynchronous compress_documents method.

        Args:
            documents: Sequence of LangChain Document objects to rerank
            query: The query string to rank documents against
            callbacks: Optional callbacks for the operation

        Returns:
            Sequence of reranked Document objects
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return asyncio.run(self.acompress_documents(documents, query, callbacks))
