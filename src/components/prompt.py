"""Prompt construction utilities for chat service.

This module provides functions to build chat prompts for different scenarios:
- Standard chat without knowledge base
- RAG (Retrieval-Augmented Generation) with knowledge base
"""

from langchain_core.prompts import ChatPromptTemplate


def create_chat_prompts(
    prompt: str | None = None,
) -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """Create chat prompt templates for both RAG and normal chat.

    Args:
        prompt: Optional custom system prompt. If None, uses default prompt.

    Returns:
        Tuple of (knowledge_prompt, normal_prompt)
        - knowledge_prompt: ChatPromptTemplate for RAG with knowledge base
        - normal_prompt: ChatPromptTemplate for standard chat without knowledge base
    """
    # Default system prompts
    system_prompt_zh = "你是一个乐于助人的助手"
    # English prompt reserved for future use
    # system_prompt_en = (
    #     "You are an assistant who helps people solve all kinds of problems."
    #     "Response in English"
    # )

    # Use custom prompt if provided, otherwise use default
    ai_info = prompt if prompt else system_prompt_zh

    # RAG-specific prompt template
    RAG_prompt_zh = """【注意：当用户向你提问，请你使用下面检索到的上下文来回答问题。如果根据检索到的上下文不能够回答问题，请你回答:'据检索到的上下文不足够回答该问题'。检索到的上下文如下：\n"""
    # English RAG prompt reserved for future use
    # RAG_prompt_en = """Note: When a user asks you a question, please answer it using the context retrieved below. If you cannot answer the question based on the retrieved context, please reply: "The retrieved context is not sufficient to answer this question." The retrieved context is as follows: \n"""

    # Knowledge base prompt template (for RAG)
    knowledge_system_prompt = f"{ai_info}{RAG_prompt_zh} {{context}}"

    knowledge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", knowledge_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    # Normal chat prompt template (without knowledge base)
    normal_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ai_info),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    return knowledge_prompt, normal_prompt
