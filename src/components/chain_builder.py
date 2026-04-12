"""Chain builder component for creating RAG and normal chat chains.

This module provides functionality to build different types of chat chains
based on knowledge base availability and configuration.
"""

import logging
from typing import Optional, Tuple

from bson import ObjectId
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

# Beanie models
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel


class ChainBuilder:
    """Builder class for creating chat chains with different configurations.

    This class handles:
    - RAG chain creation with knowledge base
    - Normal chain creation without knowledge base
    - Knowledge base metadata retrieval and context naming
    - Fallback chain creation on errors

    Attributes:
        knowledge: Knowledge instance for vector retrieval
        knowledge_prompt: ChatPromptTemplate for RAG conversations
        normal_prompt: ChatPromptTemplate for normal conversations
    """

    def __init__(
        self,
        knowledge,
        knowledge_prompt: ChatPromptTemplate,
        normal_prompt: ChatPromptTemplate,
    ):
        """Initialize the ChainBuilder.

        Args:
            knowledge: Knowledge instance for vector retrieval
            knowledge_prompt: ChatPromptTemplate for RAG conversations
            normal_prompt: ChatPromptTemplate for normal conversations
        """
        self.knowledge = knowledge
        self.knowledge_prompt = knowledge_prompt
        self.normal_prompt = normal_prompt

    def create_fallback_chain(
        self, chat, display_name: str
    ) -> Tuple[RunnableSerializable, str, bool]:
        """Create a fallback normal chat chain.

        Args:
            chat: The LLM instance
            display_name: The display name for the context

        Returns:
            Tuple of (base_chain, context_display_name, is_dict_output_for_history)
        """
        base_chain = self.normal_prompt | chat
        return base_chain, display_name, False

    async def determine_context_and_base_chain(
        self,
        chat,
        knowledge_base_id: Optional[str],
        filter_by_file_md5: Optional[list[str]],
        search_k: int,
    ) -> Tuple[str, RunnableSerializable, bool]:
        """Determine context display name and create appropriate chat chain.

        This method decides whether to create a RAG chain (with knowledge base)
        or a normal chain (without knowledge base) based on the provided parameters.

        Args:
            chat: The LLM instance
            knowledge_base_id: Optional knowledge base ID for RAG
            filter_by_file_md5: Optional list of file MD5s to filter by
            search_k: Number of documents to retrieve for RAG

        Returns:
            Tuple of (context_display_name, base_chain, is_dict_output_for_history)
            - context_display_name: Display name for the chat context
            - base_chain: The created chain
            - is_dict_output_for_history: Whether the chain outputs dict for history
        """
        context_display_name = "标准对话"
        is_dict_output_for_history = False
        base_chain: RunnableSerializable

        if knowledge_base_id and self.knowledge:
            logging.info(
                f"使用知识库: {knowledge_base_id}, 文件过滤器 MD5: {filter_by_file_md5}"
            )

            # Retrieve knowledge base metadata from MongoDB
            kb_data = await self._get_knowledge_base_data(knowledge_base_id)

            # Set context display name based on knowledge base metadata
            context_display_name = self._get_context_display_name(
                kb_data, knowledge_base_id, filter_by_file_md5
            )

            # Create RAG chain if knowledge base data is valid
            if kb_data:
                base_chain, is_dict_output_for_history = await self._create_rag_chain(
                    chat=chat,
                    kb_data=kb_data,
                    knowledge_base_id=knowledge_base_id,
                    filter_by_file_md5=filter_by_file_md5,
                    search_k=search_k,
                    context_display_name=context_display_name,
                )
            else:
                # Fallback to normal chain if KB data retrieval failed
                logging.warning(
                    f"由于无法获取知识库 {knowledge_base_id} 数据，将使用普通聊天模式。"
                )
                base_chain, context_display_name, is_dict_output_for_history = (
                    self.create_fallback_chain(chat, "标准对话")
                )
        else:
            # No knowledge base specified, use normal chat
            logging.info("不使用知识库，使用普通聊天模式。")
            base_chain, context_display_name, is_dict_output_for_history = (
                self.create_fallback_chain(chat, "标准对话")
            )

        return context_display_name, base_chain, is_dict_output_for_history

    async def _get_knowledge_base_data(
        self, knowledge_base_id: str
    ) -> Optional[dict]:
        """Retrieve knowledge base data from MongoDB.

        Args:
            knowledge_base_id: The knowledge base ID

        Returns:
            Knowledge base data as dict, or None if not found/error
        """
        if not ObjectId.is_valid(knowledge_base_id):
            return None

        logging.info(f"从 MongoDB 获取知识库: {knowledge_base_id}")
        try:
            knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(knowledge_base_id))
            if knowledge_base_doc:
                logging.info(f"从 MongoDB 成功获取知识库: {knowledge_base_id}")
                return knowledge_base_doc.model_dump(mode="json")
            else:
                logging.warning(
                    f"在 MongoDB 中未找到 ID 为 {knowledge_base_id} 的知识库文档。"
                )
                return None
        except Exception as e:
            logging.error(f"查询 MongoDB 知识库 {knowledge_base_id} 时出错: {e}")
            return None

    def _get_context_display_name(
        self,
        kb_data: Optional[dict],
        knowledge_base_id: str,
        filter_by_file_md5: Optional[list[str]],
    ) -> str:
        """Generate context display name based on knowledge base and file filter.

        Args:
            kb_data: Knowledge base data from MongoDB
            knowledge_base_id: The knowledge base ID
            filter_by_file_md5: Optional list of file MD5s to filter by

        Returns:
            Context display name string
        """
        if not kb_data:
            logging.warning(
                f"无法从 MongoDB 获取知识库 {knowledge_base_id} 的元数据。"
            )
            return "标准对话 (知识库数据错误)"

        kb_title = kb_data.get("title", "未知知识库")

        # If file filter is specified, try to find matching file
        if filter_by_file_md5:
            files_list = kb_data.get("filesList")
            if isinstance(files_list, list):
                for file_info in files_list:
                    if (
                        isinstance(file_info, dict)
                        and str(file_info.get("file_md5")) in filter_by_file_md5
                    ):
                        return f"文件：{file_info.get('file_name', '未知文件名')}"

                logging.warning(
                    f"在知识库 {knowledge_base_id} 中未找到 MD5 为 {filter_by_file_md5} 的文件，将显示知识库名称。"
                )

        return f"知识库：{kb_title}"

    async def _create_rag_chain(
        self,
        chat,
        kb_data: dict,
        knowledge_base_id: str,
        filter_by_file_md5: Optional[list[str]],
        search_k: int,
        context_display_name: str,
    ) -> Tuple[RunnableSerializable, bool]:
        """Create a RAG chain with knowledge base retrieval.

        Args:
            chat: The LLM instance
            kb_data: Knowledge base data from MongoDB
            knowledge_base_id: The knowledge base ID
            filter_by_file_md5: Optional list of file MD5s to filter by
            search_k: Number of documents to retrieve
            context_display_name: Initial context display name

        Returns:
            Tuple of (base_chain, is_dict_output_for_history)
        """
        filter_dict = None
        if filter_by_file_md5:
            filter_dict = {"source_file_md5": {"$in": filter_by_file_md5}}

        try:
            retriever = await self.knowledge.get_retriever_for_knowledge_base(
                kb_id=knowledge_base_id,
                filter_dict=filter_dict,
                search_k=search_k,
            )
            question_answer_chain = create_stuff_documents_chain(
                chat, self.knowledge_prompt
            )
            base_chain = create_retrieval_chain(retriever, question_answer_chain)
            logging.info("RAG 链创建成功。")
            return base_chain, True

        except FileNotFoundError as e:
            logging.warning(
                f"无法加载知识库向量存储 {knowledge_base_id} (可能不存在或无法访问): {e}。将退回到普通聊天模式。"
            )
            return self.create_fallback_chain(chat, "标准对话 (知识库向量错误)")

        except Exception as e:
            logging.error(
                f"获取知识库检索器或创建 RAG 链时出错 ({knowledge_base_id}): {e}",
                exc_info=True,
            )
            return self.create_fallback_chain(chat, "标准对话 (知识库错误)")
