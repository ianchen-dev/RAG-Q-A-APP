"""Stream handler component for processing chat response chunks.

This module provides functionality to handle different types of streaming chunks
from LangChain chains, including message chunks, dictionary chunks (RAG), and context data.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import BaseMessage


class StreamHandler:
    """Handles streaming response chunks from chat chains.

    This class processes different types of chunks that can be emitted during
    streaming:
    - BaseMessage chunks (from normal chat chains)
    - Dict chunks (from RAG chains with 'answer' and 'context' keys)
    - String chunks (raw string output)

    Attributes:
        serialize_document_fn: Function to serialize document objects
    """

    def __init__(self, serialize_document_fn=None):
        """Initialize the StreamHandler.

        Args:
            serialize_document_fn: Optional function to serialize document objects.
                If None, uses the default _serialize_document method.
        """
        self.serialize_document_fn = (
            serialize_document_fn if serialize_document_fn else self._serialize_document
        )

    def handle_message_chunk(self, chunk: BaseMessage) -> Optional[str]:
        """Handle a message chunk from the stream.

        Args:
            chunk: The chunk to process (expected to be BaseMessage)

        Returns:
            Content string if available, None otherwise
        """
        if hasattr(chunk, "content"):
            return chunk.content
        return None

    def handle_dict_chunk(
        self, chunk: Dict, knowledge_base_id: Optional[str]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle a dictionary chunk from the stream.

        Args:
            chunk: The dictionary chunk to process
            knowledge_base_id: The knowledge base ID for context docs

        Returns:
            Tuple of (content_piece, context_data)
            - content_piece: Answer content if available
            - context_data: Dictionary with type and data for yielding, or None
        """
        if "answer" in chunk:
            return self._handle_answer_chunk(chunk["answer"])
        elif "context" in chunk:
            return self._handle_context_chunk(chunk["context"], knowledge_base_id)
        else:
            logging.debug(f"流中接收到未处理的字典块: {chunk.keys()}")
            return None, None

    def _handle_answer_chunk(
        self, answer_part: Any
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle the answer part of a dictionary chunk.

        Args:
            answer_part: The answer content from the chunk

        Returns:
            Tuple of (content_piece, context_data)
        """
        # Sometimes RAG outputs answer as a complete string
        if isinstance(answer_part, str):
            return answer_part, None
        # Sometimes RAG outputs answer as AIMessageChunk
        elif isinstance(answer_part, BaseMessage) and hasattr(answer_part, "content"):
            return answer_part.content, None
        elif answer_part is not None:  # Avoid processing None
            logging.debug(
                f"流中 'answer' 字段的非预期类型: {type(answer_part)}"
            )
        return None, None

    def _handle_context_chunk(
        self, context_part: Any, knowledge_base_id: Optional[str]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle the context part of a dictionary chunk.

        Args:
            context_part: The context data from the chunk
            knowledge_base_id: The knowledge base ID for context docs

        Returns:
            Tuple of (content_piece, context_data)
        """
        processed_context = []

        if isinstance(context_part, list):
            for doc in context_part:
                processed_context.append(
                    self.serialize_document_fn(doc)
                )
        else:
            logging.warning(
                f"context_part is not a list as expected: {context_part}"
            )
            # Handle case where context_part is a single dict
            if isinstance(context_part, dict):
                processed_context.append(context_part)
            else:
                processed_context.append(
                    {
                        "error": "Context is not a list",
                        "original_context": str(context_part),
                    }
                )

        # Convert processed_context to formatted JSON string
        content_json_str = json.dumps(
            processed_context, indent=2, ensure_ascii=False
        )
        logging.info(
            f"发送给前端的格式化上下文JSON: {content_json_str}"
        )

        context_data = {
            "type": "tool_result",
            "data": {
                "name": "知识库检索",
                "tool_call_id": knowledge_base_id,
                "content": content_json_str,
            },
        }
        return None, context_data

    def _serialize_document(self, doc: Any) -> Dict:
        """Serialize a document object to a dictionary.

        This is the default serialization method. Can be overridden by
        providing a custom serialize_document_fn in __init__.

        Args:
            doc: Document object to serialize

        Returns:
            Dictionary representation of the document
        """
        try:
            # Use model_dump() for dictionary representation
            doc_dict = doc.model_dump(exclude_none=True)
            return doc_dict
        except AttributeError as e:
            # Fallback if Document doesn't have model_dump method
            logging.warning(
                f"尝试对 Document 对象调用 model_dump() 时出错: {e}. "
                f"文档内容: {doc}. 将尝试手动提取。"
            )
            if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                return {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            else:
                return {
                    "error": "Invalid document structure for fallback",
                    "original_doc": str(doc),
                }
        except Exception as e:
            logging.error(
                f"序列化 Document 对象时发生未知错误: {e}. 文档内容: {doc}"
            )
            return {
                "error": "Unknown serialization error",
                "original_doc": str(doc),
            }

    def process_stream_chunk(
        self, chunk: Any, knowledge_base_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Process a stream chunk based on its type.

        This is the main entry point for processing chunks from the stream.
        It automatically detects the chunk type and routes to the appropriate handler.

        Args:
            chunk: The chunk to process (can be BaseMessage, dict, or str)
            knowledge_base_id: Optional knowledge base ID for context docs

        Returns:
            Tuple of (content_piece, context_data)
            - content_piece: Extracted content if available
            - context_data: Context data dict for yielding, or None
        """
        content_piece = None
        context_data = None

        if isinstance(chunk, BaseMessage):
            # Normal chain (prompt | llm) outputs AIMessageChunk
            content_piece = self.handle_message_chunk(chunk)
        elif isinstance(chunk, dict):
            # RAG chain outputs dict {'answer': ..., 'context': ...}
            content_piece, context_data = self.handle_dict_chunk(
                chunk, knowledge_base_id
            )
        elif isinstance(chunk, str):
            # Direct string output
            content_piece = chunk
        else:
            logging.warning(
                f"流中接收到未知类型的块: {type(chunk)}, 内容: {chunk}"
            )

        return content_piece, context_data
