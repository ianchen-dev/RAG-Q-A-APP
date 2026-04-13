import logging
import os
from typing import (
    Any,
    AsyncIterable,
    Dict,
    Optional,
    Tuple,
    Union,
)

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    ConfigurableFieldSpec,  # f-流式输出
    RunnableConfig,  # f-流式输出
    RunnableLambda,
    RunnableSerializable,  # f-流式输出
)
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)  # f-历史会话-对话引用历史会话

# components
from src.components.chain_builder import ChainBuilder
from src.components.chat_history import ChatHistoryManager
from src.components.prompt import create_chat_prompts
from src.components.stream_handler import StreamHandler

# utils
from src.components.kb import KnowledgeManager
from src.utils.llm_modle import get_llms

logger = logging.getLogger(__name__)


class ChatSev:
    # _chat_history = ChatMessageHistory()  # 对话历史

    def __init__(
        self,
        knowledge: Optional[KnowledgeManager] = None,
        prompt: str | None = None,
        chat_history_max_length: Optional[int] = 1,
    ):
        self.knowledge = knowledge  # Store the initialized Knowledge instance
        # self.chat_history_max_length = chat_history_max_length # 暂时注释掉，因为 MongoDB History 不直接限制长度

        # 从环境变量读取 MongoDB 配置
        # 使用包含认证信息的 MONGODB_URL
        self.mongo_connection_string = os.getenv("MONGODB_URL")
        if not self.mongo_connection_string:
            raise ValueError(
                "错误：环境变量 MONGODB_URL 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )
        self.mongo_database_name = os.getenv("MONGO_DB_NAME")
        if not self.mongo_database_name:
            raise ValueError(
                "错误：环境变量 MONGO_DB_NAME 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )
        # 使用 .env 中定义的集合名称
        self.mongo_collection_name = os.getenv("MONGODB_COLLECTION_NAME_CHATHISTORY")
        if not self.mongo_collection_name:
            raise ValueError(
                "错误：环境变量 MONGODB_COLLECTION_NAME_CHATHISTORY 未设置或为空。请检查您的 .env 文件或系统环境变量。"
            )
        self.prompt = prompt  # f-提示词功能-传入自定义提示词
        # Create chat prompts using the extracted function
        self.knowledge_prompt, self.normal_prompt = create_chat_prompts(prompt)

        # Initialize chat history manager
        self.chat_history_manager = ChatHistoryManager(
            mongo_connection_string=self.mongo_connection_string,
            mongo_database_name=self.mongo_database_name,
            mongo_collection_name=self.mongo_collection_name,
            normal_prompt=self.normal_prompt,
            knowledge_prompt=self.knowledge_prompt,
        )

        # Initialize chain builder
        self.chain_builder = ChainBuilder(
            knowledge=knowledge,
            knowledge_prompt=self.knowledge_prompt,
            normal_prompt=self.normal_prompt,
        )

        # Initialize stream handler
        self.stream_handler = StreamHandler()

    def get_session_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """根据 session_id 获取 MongoDB 聊天记录实例"""
        return self.chat_history_manager.get_session_chat_history(session_id)

    def _create_fallback_chain(
        self, chat, display_name: str
    ) -> Tuple[RunnableSerializable, str, bool]:
        """Create a fallback normal chat chain with the specified display name.

        Delegates to ChainBuilder.

        Args:
            chat: The LLM instance
            display_name: The display name for the context

        Returns:
            Tuple of (base_chain, context_display_name, is_dict_output_for_history)
        """
        return self.chain_builder.create_fallback_chain(chat, display_name)

    async def _determine_context_and_base_chain(
        self,
        api_key: Optional[str],
        supplier: str,
        model: str,
        knowledge_base_id: Optional[str],
        filter_by_file_md5: Optional[list[str]],
        search_k: int,
        max_length: Optional[int],
        temperature: float,
    ) -> Tuple[str, RunnableSerializable, bool]:
        """辅助函数：确定上下文显示名称、基础链以及输出是否为字典以供历史记录使用。

        Delegates to ChainBuilder after creating the LLM instance.

        Returns:
            Tuple[str, RunnableSerializable, bool]: (上下文名称, 基础链, 是否为字典输出)
        """
        chat = get_llms(
            supplier=supplier,
            model=model,
            api_key=api_key,
            max_length=max_length,
            temperature=temperature,
        )

        return await self.chain_builder.determine_context_and_base_chain(
            chat=chat,
            knowledge_base_id=knowledge_base_id,
            filter_by_file_md5=filter_by_file_md5,
            search_k=search_k,
        )

    def _serialize_document(self, doc) -> Dict:
        """Serialize a document object to a dictionary.

        Delegates to StreamHandler.

        Args:
            doc: Document object to serialize

        Returns:
            Dictionary representation of the document
        """
        return self.stream_handler._serialize_document(doc)

    def _handle_message_chunk(self, chunk) -> Optional[str]:
        """Handle a message chunk from the stream.

        Delegates to StreamHandler.

        Args:
            chunk: The chunk to process (expected to be BaseMessage)

        Returns:
            Content string if available, None otherwise
        """
        return self.stream_handler.handle_message_chunk(chunk)

    def _handle_dict_chunk(
        self, chunk: Dict, knowledge_base_id: Optional[str]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Handle a dictionary chunk from the stream.

        Delegates to StreamHandler.

        Args:
            chunk: The dictionary chunk to process
            knowledge_base_id: The knowledge base ID for context docs

        Returns:
            Tuple of (content_piece, context_data)
            - content_piece: Answer content if available
            - context_data: Dictionary with type and data for yielding, or None
        """
        return self.stream_handler.handle_dict_chunk(chunk, knowledge_base_id)

    # f-流式输出
    async def stream_chat(
        self,
        question: str,
        api_key: Optional[str],
        supplier: str,
        model: str,
        session_id: str,
        knowledge_base_id: Optional[str] = None,
        filter_by_file_md5: Optional[str] = None,
        search_k: int = 3,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
    ) -> AsyncIterable[Dict[str, Any]]:  # 返回结构化的字典流
        """
        f-流式输出-异步执行聊天调用，并流式返回结果。
        Yields:
            字典，包含 'type' ('context', 'chunk', 'error') 和 'data'。
        """
        try:
            # 1. 确定上下文和基础链
            (
                context_display_name,
                base_chain,
                is_dict_output_for_history,
            ) = await self._determine_context_and_base_chain(
                api_key,
                supplier,
                model,
                knowledge_base_id,
                filter_by_file_md5,
                search_k,
                max_length,
                temperature,
            )

            # 1.f-流式输出-发送上下文信息作为流的第一个元素
            # yield {"type": "context", "data": context_display_name}

            # 2.f-历史会话-包装历史记录管理
            # RunnableWithMessageHistory 会自动处理输入和历史消息，并将 base_chain 的输出传递出去
            history_output_key = "answer" if is_dict_output_for_history else None
            if history_output_key == "answer":
                yield {
                    "type": "tool_call",
                    "data": {
                        "name": "知识库检索",
                        "args": context_display_name,
                        "id": knowledge_base_id,
                    },
                }
            else:
                yield {"type": "context", "data": context_display_name}  # 普通对话

            chain_with_history = RunnableWithMessageHistory(
                base_chain,
                self.get_session_chat_history,  # f-历史会话-获取会话历史-MongoDBChatMessageHistory
                input_messages_key="input",  # base_chain 需要 'input'
                history_messages_key="chat_history",  # prompt 需要 'chat_history'
                output_messages_key=history_output_key,  # <--- 使用动态键
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="session_id",
                        annotation=str,
                        name="Session ID",
                        description="Unique identifier for the chat session.",
                        default="",
                        is_shared=True,
                    )
                ],
            )

            # 3. f-流式输出-配置并调用 astream
            config: RunnableConfig = {"configurable": {"session_id": session_id}}
            logging.info(
                f"使用 session_id: {session_id} 调用流式链 ({'RAG' if knowledge_base_id and self.knowledge else 'Normal'})... 上下文: {context_display_name}"
            )
            # f-流式输出-Chain.astream()
            stream_iterator = chain_with_history.astream(
                {"input": question}, config=config
            )

            # 4. f-流式输出-处理流式块
            async for chunk in stream_iterator:
                # Use StreamHandler to process the chunk
                content_piece, context_data = self.stream_handler.process_stream_chunk(
                    chunk, knowledge_base_id
                )

                # 发送 context_data if available
                if context_data:
                    yield context_data
                # 发送 chunk
                if content_piece:  # 仅当提取到有效内容时才发送 chunk
                    yield {"type": "chunk", "data": content_piece}

        except Exception as e:
            logging.error(
                f"流式处理时发生错误 (session_id: {session_id}): {e}", exc_info=True
            )
            # 在流中发送错误信息
            yield {"type": "error", "data": f"处理请求时发生错误: {e}"}

    # 废弃-保留 invoke 方法，以防需要非流式接口
    # 注意：当前的 invoke 实现依赖于旧的链结构和 streaming_parse，需要更新以匹配新逻辑
    async def invoke(
        self,
        question: str,
        api_key: Optional[str],
        supplier: str,
        model: str,
        session_id: str,
        knowledge_base_id: Optional[str] = None,
        filter_by_file_md5: Optional[str] = None,
        search_k: int = 3,
        max_length=None,
        temperature=0.8,
    ) -> Dict[str, Any]:  # 返回字典
        """
        废弃-保留 invoke 方法，以防需要非流式接口
        (注意：此方法可能需要更新或移除，当前为旧版非流式实现)
        异步执行聊天调用，并一次性返回结果。
        """
        logging.warning("调用了旧版 invoke 方法，考虑切换到 stream_chat。")
        (
            context_display_name,
            base_chain,
            is_dict_output_for_history,
        ) = await self._determine_context_and_base_chain(
            api_key,
            supplier,
            model,
            knowledge_base_id,
            filter_by_file_md5,
            search_k,
            max_length,
            temperature,
        )

        # 需要重新构建适用于 ainvoke 的链，因为 base_chain 输出格式可能不同
        # 普通链: prompt | llm -> AIMessage
        # RAG链: retrieval_chain -> Dict{'answer':..., 'context':...}

        # 这里需要一个转换器，确保最终输出是期望的字典格式，或者调整后续处理逻辑
        # 例如，如果 base_chain 是普通链，需要包装输出

        def format_output(input_data: Union[BaseMessage, Dict]) -> Dict[str, Any]:
            if isinstance(input_data, BaseMessage):
                return {
                    "answer": input_data.content,
                    "context_display_name": context_display_name,
                }
            elif isinstance(input_data, dict):
                input_data["context_display_name"] = context_display_name
                return input_data
            else:
                # Fallback or raise error
                return {
                    "answer": str(input_data),
                    "context_display_name": context_display_name,
                }

        final_chain = base_chain | RunnableLambda(format_output)

        chain_with_history = RunnableWithMessageHistory(
            final_chain,  # 使用调整后的链
            self.get_session_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            # 对于 ainvoke, history 会自动管理，我们关心 final_chain 的输出
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for the chat session.",
                    default="",
                    is_shared=True,
                )
            ],
        )

        config = {"configurable": {"session_id": session_id}}
        logging.info(
            f"使用 session_id: {session_id} 调用非流式链 ({'RAG' if knowledge_base_id and self.knowledge else 'Normal'})... 上下文: {context_display_name}"
        )

        try:
            # 使用 ainvoke 进行异步调用
            result = await chain_with_history.ainvoke(
                {"input": question}, config=config
            )
            # format_output 已添加 context_display_name，无需再次添加
            return result  # format_output 确保返回字典
        except Exception as e:
            logging.error(f"执行非流式 invoke 时出错: {e}", exc_info=True)
            # 返回错误信息字典
            return {
                "error": f"处理请求时发生错误: {e}",
                "context_display_name": context_display_name,
            }

    def clear_history(self, session_id: str) -> None:
        """清除指定 session_id 的历史信息"""
        history = self.get_session_chat_history(session_id)
        history.clear()
        logging.info(f"已清除 session_id 为 {session_id} 的 MongoDB 历史记录")

    def get_history_message(self, session_id: str) -> list:
        """获取指定 session_id 的历史信息"""
        history = self.get_session_chat_history(session_id)
        logging.info(f"获取 session_id 为 {session_id} 的 MongoDB 历史消息")
        # 返回其 messages 属性，需要注意 MongoDBChatMessageHistory 的 messages 可能是内部表示
        # Langchain 标准接口是 get_messages() 方法
        try:
            # 假设 MongoDBChatMessageHistory 实现了 get_messages 或 messages 属性
            # 优先使用 get_messages()
            if hasattr(history, "get_messages") and callable(history.get_messages):
                return history.get_messages()
            elif hasattr(history, "messages"):  # 备选方案
                return history.messages
            else:
                logging.warning(
                    f"无法从 MongoDBChatMessageHistory (session: {session_id}) 获取消息列表。"
                )
                return []
        except Exception as e:
            logging.error(f"获取历史消息时出错 (session: {session_id}): {e}")
            return []
