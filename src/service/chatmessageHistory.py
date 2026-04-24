# 旧方案
# 2025年4月9日
# 使用 ChatMessageHistory ，内存存储历史信息的方式


import logging
from typing import Iterable, Optional

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import Output

from src.components.kb import KnowledgeManager

# utils
from src.components.llm_provider import get_llms


class ChatSev:
    _chat_history = ChatMessageHistory()  # 对话历史

    def __init__(
        self, knowledge: Optional[KnowledgeManager], chat_history_max_length: Optional[int] = 8
    ):
        self.knowledge: Optional[KnowledgeManager] = knowledge
        self.chat_history_max_length: int = (
            chat_history_max_length if chat_history_max_length is not None else 8
        )

        self.knowledge_prompt = None  # 问答模板
        self.normal_prompt = None  # 正常模板
        self.create_chat_prompt()  # 创建聊天模板

    def create_chat_prompt(self) -> None:
        ai_info = "你叫超级无敌霸王龙🦖，一个帮助人们解答各种问题的助手。"

        # AI系统prompt
        knowledge_system_prompt = (
            f"{ai_info} 当用户向你提问，请你使用下面检索到的上下文来回答问题。如果检索到的上下文中没有问题的答案，请你直接回答不知道。检索到的上下文如下：\n\n"
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

    @staticmethod
    def streaming_parse(chunks: Iterable[AIMessageChunk]) -> list[AddableDict]:
        """统一模型的输出格式，将模型的输出存储到字典answer的value中"""
        for chunk in chunks:
            yield AddableDict({"answer": chunk.content})

    def get_chain(
        self,
        api_key: Optional[str],
        collection: str,
        supplier: str,
        model: str,
        max_length: int,
        temperature: float = 0.8,
    ) -> RunnableWithMessageHistory:
        """获取聊天链"""
        # chat = ChatOpenAI(model=model, max_tokens=max_length, temperature=temperature)
        chat = get_llms(
            supplier=supplier,
            model=model,
            api_key=api_key,
            max_length=max_length,
            temperature=temperature,
        )

        # 统一返回逻辑
        rag_chain = (
            self.normal_prompt | chat | self.streaming_parse
            if collection is None
            else create_retrieval_chain(
                self.knowledge.get_retrievers(collection),
                create_stuff_documents_chain(chat, self.knowledge_prompt),
            )
        )
        logging.info("Chat history content: %s", self._chat_history.messages)

        return RunnableWithMessageHistory(
            rag_chain,  # 传入聊天链
            # lambda session_id: self._chat_history,
            get_session_history=self.get_session_chat_history,  # 传入历史信息
            input_messages_key="input",  # 输入信息的键名
            history_messages_key="chat_history",  # 历史信息的键名
            output_messages_key="answer",  # 输出答案
        )

    def get_session_chat_history(self):
        return self._chat_history

    def invoke(
        self,
        question: str,
        api_key: Optional[str],
        collection: Optional[str],
        supplier: str,
        model: str,
        max_length=None,
        temperature=0.8,
    ) -> Output:
        """
        :param question: 用户提出的问题 例如: '请问你是谁？'
        :param collection: 知识库文件名称 例如:'人事管理流程.docx'
        :param model: 使用模型,默认为 'gpt-3.5-turbo'
        :param max_length: 数据返回最大长度
        :param temperature: 数据温度值
        """
        logging.info("Chat history content: %s", self._chat_history.messages)
        return self.get_chain(
            api_key, collection, supplier, model, max_length, temperature
        ).invoke({"input": question})

    def clear_history(self) -> None:
        """清除历史信息"""
        self._chat_history.clear()

    def get_history_message(self) -> list:
        """获取历史信息"""
        return self._chat_history.messages
