import json
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

import redis.asyncio as aioredis  # 导入 aioredis
from bson import ObjectId
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableFieldSpec,  # f-流式输出
    RunnableConfig,  # f-流式输出
    RunnableLambda,
    RunnableSerializable,  # f-流式输出
)
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)  # f-历史会话-对话引用历史会话
from langchain_mongodb.chat_message_histories import (
    MongoDBChatMessageHistory,  # f-历史会话-持久化会话历史数据
)

# Redis 缓存
from src.config.Redis import get_redis_client

# Beanie模型
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel
from src.service.knowledgeSev import (  # 导入缓存设置函数和前缀
    KB_CACHE_PREFIX,
    _set_kb_cache,
)
from src.utils.Knowledge import Knowledge

# utils
from src.utils.llm_modle import get_llms

logger = logging.getLogger(__name__)


class ChatSev:
    # 废弃-移除类级别的内存历史记录实例
    # _chat_history = ChatMessageHistory()  # 对话历史

    def __init__(
        self,
        knowledge: Optional[Knowledge] = None,
        prompt: str | None = None,
        chat_history_max_length: Optional[int] = 8,
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
        self.knowledge_prompt = None  # 问答模板
        self.normal_prompt = None  # 正常模板
        self.create_chat_prompt()  # 创建聊天模板

    def create_chat_prompt(self) -> None:
        ai_info = self.prompt if self.prompt else "你是一个帮助人们解答各种问题的助手。"

        # 知识库prompt--system
        knowledge_system_prompt = (
            f"{ai_info} 【注意：当用户向你提问，请你使用下面检索到的上下文来回答问题。如果检索到的上下文中没有问题的答案，请你回答:'根据检索到的上下文，我无法准确回答这个问题'。检索到的上下文如下：\n\n"
            "{context}"
        )

        self.knowledge_prompt = ChatPromptTemplate.from_messages(  # 知识库prompt
            [
                ("system", knowledge_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # 没有指定知识库的模板的AI系统模板
        self.normal_prompt = ChatPromptTemplate.from_messages(  # 正常prompt
            [
                ("system", ai_info),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

    def get_session_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """根据 session_id 获取 MongoDB 聊天记录实例"""
        logging.info(f"获取 session_id 为 {session_id} 的 MongoDB 聊天记录")
        return MongoDBChatMessageHistory(
            connection_string=self.mongo_connection_string,
            session_id=session_id,
            database_name=self.mongo_database_name,
            collection_name=self.mongo_collection_name,
        )

    async def _determine_context_and_base_chain(
        self,
        api_key: Optional[str],
        supplier: str,
        model: str,
        knowledge_base_id: Optional[str],
        filter_by_file_md5: Optional[str],
        search_k: int,
        max_length: Optional[int],
        temperature: float,
    ) -> Tuple[str, RunnableSerializable, bool]:
        """辅助函数：确定上下文显示名称、基础链以及输出是否为字典以供历史记录使用。

        Returns:
            Tuple[str, RunnableSerializable, bool]: (上下文名称, 基础链, 是否为字典输出)
        """
        context_display_name = "标准对话"
        is_dict_output_for_history = False
        chat = get_llms(
            supplier=supplier,
            model=model,
            api_key=api_key,
            max_length=max_length,
            temperature=temperature,
        )
        base_chain: RunnableSerializable

        if knowledge_base_id and self.knowledge:
            logging.info(
                f"使用知识库: {knowledge_base_id}, 文件过滤器 MD5: {filter_by_file_md5}"
            )
            kb_data = None  # 存储从缓存或 DB 获取的知识库数据

            # 尝试从 Redis 缓存获取知识库元数据-展示上下文信息
            try:
                if ObjectId.is_valid(knowledge_base_id):
                    redis = get_redis_client()
                    cache_key = f"{KB_CACHE_PREFIX}{knowledge_base_id}"
                    cached_data_str = await redis.get(cache_key)
                    if cached_data_str:
                        kb_data = json.loads(cached_data_str)  # 反序列化 JSON
                        logger.info(f"从 Redis 缓存命中知识库: {knowledge_base_id}")
                    else:
                        logger.info(f"Redis 缓存未命中知识库: {knowledge_base_id}")
                else:
                    logger.warning(
                        f"提供的 knowledge_base_id 无效 (格式错误): {knowledge_base_id}"
                    )

            except aioredis.RedisError as e:
                logger.error(
                    f"访问 Redis 缓存知识库 {knowledge_base_id} 时出错: {e}. 将尝试从 MongoDB 回退。"
                )
                kb_data = None  # 确保在出错时重置
            except json.JSONDecodeError as e:
                logger.error(
                    f"解析 Redis 缓存中的知识库 {knowledge_base_id} 数据时出错: {e}. 将尝试从 MongoDB 回退。"
                )
                kb_data = None  # 确保在出错时重置
            except Exception as e:
                logger.error(
                    f"读取或解析 Redis 缓存 {knowledge_base_id} 时发生未知错误: {e}",
                    exc_info=True,
                )
                kb_data = None

            # 如果缓存未命中或出错，则从 MongoDB 回退
            if kb_data is None and ObjectId.is_valid(knowledge_base_id):
                logger.info(f"尝试从 MongoDB 获取知识库: {knowledge_base_id}")
                try:
                    knowledge_base_doc = await KnowledgeBaseModel.get(
                        ObjectId(knowledge_base_id)
                    )
                    if knowledge_base_doc:
                        logger.info(f"从 MongoDB 成功获取知识库: {knowledge_base_id}")
                        # 将 Beanie 文档转换为字典，以便后续逻辑统一处理
                        kb_data = knowledge_base_doc.model_dump(mode="json")
                        # 尝试写回缓存 (缓存自愈)
                        await _set_kb_cache(
                            knowledge_base_doc
                        )  # 使用 knowledgeSev 中的辅助函数
                    else:
                        logger.warning(
                            f"在 MongoDB 中未找到 ID 为 {knowledge_base_id} 的知识库文档。"
                        )
                except Exception as e:
                    logger.error(f"查询 MongoDB 知识库 {knowledge_base_id} 时出错: {e}")

            # 使用获取到的 kb_data (来自缓存或 DB) 设置上下文名称
            if kb_data:
                kb_title = kb_data.get("title", "未知知识库")
                if filter_by_file_md5:
                    file_found = False
                    # 确保 filesList 存在且是列表
                    files_list = kb_data.get("filesList")
                    if isinstance(files_list, list):
                        for file_info in files_list:
                            # 确保 file_info 是字典且包含 file_md5
                            if isinstance(file_info, dict) and str(
                                file_info.get("file_md5")
                            ) == str(filter_by_file_md5):
                                context_display_name = (
                                    f"文件：{file_info.get('file_name', '未知文件名')}"
                                )
                                file_found = True
                                break
                    if not file_found:
                        logger.warning(
                            f"在知识库 {knowledge_base_id} (来自 {'缓存' if cached_data_str else 'DB'}) 中未找到 MD5 为 {filter_by_file_md5} 的文件，将显示知识库名称。"
                        )
                        context_display_name = f"知识库：{kb_title}"
                else:
                    context_display_name = f"知识库：{kb_title}"
            else:
                # 如果缓存和 DB 都获取失败
                logger.warning(
                    f"无法从缓存或 MongoDB 获取知识库 {knowledge_base_id} 的元数据。"
                )
                context_display_name = "标准对话 (知识库数据错误)"

            # --- RAG 链创建逻辑 (基本不变，依赖 kb_data 是否有效来决定是否创建 RAG) ---
            if kb_data:  # 只有成功获取到数据才尝试创建 RAG 链
                filter_dict = None
                if filter_by_file_md5:
                    filter_dict = {"source_file_md5": str(filter_by_file_md5)}

                try:
                    retriever = await self.knowledge.get_retriever_for_knowledge_base(
                        kb_id=knowledge_base_id,
                        filter_dict=filter_dict,
                        search_k=search_k,
                    )
                    question_answer_chain = create_stuff_documents_chain(
                        chat, self.knowledge_prompt
                    )
                    base_chain = create_retrieval_chain(
                        retriever, question_answer_chain
                    )
                    logging.info("RAG 链创建成功。")
                    is_dict_output_for_history = True
                except FileNotFoundError as e:
                    logging.warning(
                        f"无法加载知识库向量存储 {knowledge_base_id} (可能不存在或无法访问): {e}。将退回到普通聊天模式。"
                    )
                    base_chain = self.normal_prompt | chat
                    context_display_name = "标准对话 (知识库向量错误)"
                    is_dict_output_for_history = False
                except Exception as e:
                    logging.error(
                        f"获取知识库检索器或创建 RAG 链时出错 ({knowledge_base_id}): {e}",
                        exc_info=True,
                    )
                    base_chain = self.normal_prompt | chat
                    context_display_name = "标准对话 (知识库错误)"
                    is_dict_output_for_history = False
            else:
                # 如果 kb_data 获取失败，直接使用普通链
                logging.warning(
                    f"由于无法获取知识库 {knowledge_base_id} 数据，将使用普通聊天模式。"
                )
                base_chain = self.normal_prompt | chat
                is_dict_output_for_history = False

        else:  # 不使用知识库的情况
            logging.info("不使用知识库，使用普通聊天模式。")
            base_chain = self.normal_prompt | chat
            is_dict_output_for_history = False

        return context_display_name, base_chain, is_dict_output_for_history

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
                content_piece = ""  # 返回的内容块
                context_part = ""  # 返回的上下文块
                # --- 核心处理逻辑 ---
                # 根据chunk类型判断链的类型，进而得知其对应的输出数据结构，解析到content_piece
                if isinstance(
                    chunk, BaseMessage
                ):  # 普通链 (prompt | llm) 输出 AIMessageChunk
                    if hasattr(chunk, "content"):
                        content_piece = chunk.content
                elif isinstance(
                    chunk, dict
                ):  # RAG 链 (retrieval_chain) 输出字典 {'answer': ..., 'context': ...}
                    # RunnableWithMessageHistory 可能进一步包装，但通常会传递 base_chain 的输出
                    if "answer" in chunk:
                        answer_part = chunk["answer"]
                        # 判断answer_part的类型，进而得知其对应的输出数据结构，解析到content_piece
                        # 有时 RAG 输出的 answer 是完整字符串
                        if isinstance(answer_part, str):
                            content_piece = answer_part
                        # 有时 RAG 输出的 answer 是 AIMessageChunk
                        elif isinstance(answer_part, BaseMessage) and hasattr(
                            answer_part, "content"
                        ):
                            content_piece = answer_part.content
                        elif answer_part is not None:  # 避免处理 None
                            logging.debug(
                                f"流中 'answer' 字段的非预期类型: {type(answer_part)}"
                            )
                    #
                    elif "context" in chunk:
                        # 处理 context 块，例如发送 'type': 'context_docs'
                        context_part = chunk["context"]
                        # logging.info(f"流中接收到原始上下文块: {context_part}") # 可以取消注释此行以调试原始数据

                        processed_context = []
                        if isinstance(context_part, list):
                            for doc in context_part:
                                try:
                                    # 使用 model_dump() 获取字典表示，而不是 model_dump_json()
                                    # exclude_none=True 可以使输出更简洁，不包含值为 None 的字段
                                    doc_dict = doc.model_dump(exclude_none=True)
                                    processed_context.append(doc_dict)
                                except AttributeError as e:
                                    # 如果 Document 对象没有 model_dump 方法 (例如，如果它不是 Pydantic 模型)
                                    # 或者发生其他与序列化相关的错误，则记录警告并尝试回退
                                    logging.warning(
                                        f"尝试对 Document 对象调用 model_dump() 时出错: {e}. 文档内容: {doc}. 将尝试手动提取。"
                                    )
                                    if hasattr(doc, "page_content") and hasattr(
                                        doc, "metadata"
                                    ):
                                        processed_context.append(
                                            {
                                                "page_content": doc.page_content,
                                                "metadata": doc.metadata,
                                            }
                                        )
                                    else:
                                        processed_context.append(
                                            {
                                                "error": "Invalid document structure for fallback",
                                                "original_doc": str(doc),
                                            }
                                        )
                                except Exception as e:
                                    logging.error(
                                        f"序列化 Document 对象时发生未知错误: {e}. 文档内容: {doc}"
                                    )
                                    processed_context.append(
                                        {
                                            "error": "Unknown serialization error",
                                            "original_doc": str(doc),
                                        }
                                    )
                        else:
                            logging.warning(
                                f"context_part is not a list as expected: {context_part}"
                            )
                            # 根据需要，可以决定在这种情况下 processed_context 应该是什么
                            # 例如，如果 context_part 本身就是单个字典，可以直接添加，或者包装在列表中
                            if isinstance(
                                context_part, dict
                            ):  # 简单处理 context_part 是单个字典的情况
                                processed_context.append(context_part)
                            else:
                                processed_context.append(
                                    {
                                        "error": "Context is not a list",
                                        "original_context": str(context_part),
                                    }
                                )

                        # 将 processed_context (字典列表) 转换为格式化的 JSON 字符串
                        content_json_str = json.dumps(
                            processed_context, indent=2, ensure_ascii=False
                        )
                        logging.info(
                            f"发送给前端的格式化上下文JSON: {content_json_str}"
                        )

                        yield {
                            "type": "tool_result",
                            "data": {
                                "name": "知识库检索",
                                "tool_call_id": knowledge_base_id,
                                "content": content_json_str,
                            },
                        }

                    else:
                        logging.debug(f"流中接收到未处理的字典块: {chunk.keys()}")
                elif isinstance(chunk, str):  # 兼容直接输出字符串的 Runnable
                    content_piece = chunk
                else:
                    logging.warning(
                        f"流中接收到未知类型的块: {type(chunk)}, 内容: {chunk}"
                    )
                # --- 结束核心处理逻辑 ---
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
