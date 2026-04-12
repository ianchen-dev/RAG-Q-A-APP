"""Chat history management component.

This module provides a manager class for handling chat history operations,
including MongoDB integration, history retrieval, and chain creation with history support.
"""

import logging
from typing import Tuple

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


class ChatHistoryManager:
    """Manages chat history operations for the chat service.

    This class handles:
    - MongoDB chat history connection and retrieval
    - Chain creation with fallback mechanisms
    - Context determination for different chat scenarios

    Attributes:
        mongo_connection_string: MongoDB connection string
        mongo_database_name: Database name in MongoDB
        mongo_collection_name: Collection name for chat history
        normal_prompt: ChatPromptTemplate for normal conversations
        knowledge_prompt: ChatPromptTemplate for RAG conversations
    """

    def __init__(
        self,
        mongo_connection_string: str,
        mongo_database_name: str,
        mongo_collection_name: str,
        normal_prompt: ChatPromptTemplate,
        knowledge_prompt: ChatPromptTemplate,
    ):
        """Initialize the ChatHistoryManager.

        Args:
            mongo_connection_string: MongoDB connection string
            mongo_database_name: Database name in MongoDB
            mongo_collection_name: Collection name for chat history
            normal_prompt: ChatPromptTemplate for normal conversations
            knowledge_prompt: ChatPromptTemplate for RAG conversations
        """
        self.mongo_connection_string = mongo_connection_string
        self.mongo_database_name = mongo_database_name
        self.mongo_collection_name = mongo_collection_name
        self.normal_prompt = normal_prompt
        self.knowledge_prompt = knowledge_prompt

    def get_session_chat_history(
        self, session_id: str
    ) -> BaseChatMessageHistory:
        """Get MongoDB chat history instance for a session.

        Args:
            session_id: The session identifier

        Returns:
            MongoDBChatMessageHistory instance for the session
        """
        logging.info(f"获取 session_id 为 {session_id} 的 MongoDB 聊天记录")
        return MongoDBChatMessageHistory(
            connection_string=self.mongo_connection_string,
            session_id=session_id,
            database_name=self.mongo_database_name,
            collection_name=self.mongo_collection_name,
        )

    def create_fallback_chain(
        self, chat, display_name: str
    ) -> Tuple[RunnableSerializable, str, bool]:
        """Create a fallback normal chat chain.

        This is used when knowledge base retrieval fails or is not available.

        Args:
            chat: The LLM instance
            display_name: The display name for the context

        Returns:
            Tuple of (base_chain, context_display_name, is_dict_output_for_history)
        """
        base_chain = self.normal_prompt | chat
        return base_chain, display_name, False
