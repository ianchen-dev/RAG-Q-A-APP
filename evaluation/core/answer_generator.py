"""
答案生成器模块
集成ChatSev和Knowledge实现RAG答案生成
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_schema import EvaluationConfig
from src.models.knowledgeBase import KnowledgeBase
from src.service.ChatSev import ChatSev
from src.utils.embedding import get_embedding
from src.utils.Knowledge import Knowledge


class AnswerGenerator:
    """RAG答案生成器，集成ChatSev和Knowledge"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.knowledge: Optional[Knowledge] = None
        self.chat_service: Optional[ChatSev] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """初始化知识库和聊天服务"""
        self.logger.info("初始化答案生成器...")

        # 如果是知识库模式，初始化Knowledge实例
        if (
            self.config.dataset.type == "knowledge_base"
            and self.config.dataset.knowledge_base
            and self.config.knowledge_config
        ):
            await self._initialize_knowledge()

        # 初始化ChatSev
        self.chat_service = ChatSev(knowledge=self.knowledge)
        self.logger.info("答案生成器初始化完成")

    async def _initialize_knowledge(self):
        """初始化Knowledge实例"""
        try:
            kb_id = self.config.dataset.knowledge_base.kb_id
            self.logger.info(f"初始化知识库: {kb_id}")

            # 从数据库获取知识库配置
            kb = await KnowledgeBase.get(kb_id)
            if not kb:
                raise ValueError(f"知识库 {kb_id} 不存在")

            if not kb.embedding_config:
                raise ValueError(f"知识库 {kb_id} 缺少embedding配置")

            # 初始化embedding
            embedding = get_embedding(
                kb.embedding_config.embedding_supplier,
                kb.embedding_config.embedding_model,
                kb.embedding_config.embedding_apikey,
            )

            # 构建重排序配置
            remote_rerank_config = None
            if (
                self.config.knowledge_config.reranker_config
                and self.config.knowledge_config.reranker_config.use_reranker
                and self.config.knowledge_config.reranker_config.reranker_type
                == "remote"
                and self.config.knowledge_config.reranker_config.remote_rerank_config
            ):
                remote_rerank_config = {
                    "api_key": self.config.knowledge_config.reranker_config.remote_rerank_config.api_key,
                    "model": self.config.knowledge_config.reranker_config.remote_rerank_config.model,
                }

            # 初始化Knowledge实例
            self.knowledge = Knowledge(
                _embeddings=embedding,
                splitter="hybrid",
                use_bm25=self.config.knowledge_config.use_bm25,
                bm25_k=self.config.knowledge_config.bm25_k,
                use_reranker=self.config.knowledge_config.reranker_config.use_reranker
                if self.config.knowledge_config.reranker_config
                else False,
                reranker_type=self.config.knowledge_config.reranker_config.reranker_type
                if self.config.knowledge_config.reranker_config
                else "remote",
                remote_rerank_config=remote_rerank_config,
                rerank_top_n=self.config.knowledge_config.reranker_config.rerank_top_n
                if self.config.knowledge_config.reranker_config
                else 3,
            )

            self.logger.info(f"知识库 {kb_id} 初始化成功")

        except Exception as e:
            self.logger.error(f"初始化知识库失败: {e}", exc_info=True)
            raise

    async def generate_answers(self, questions: List[str]) -> List[str]:
        """为问题列表生成答案

        Args:
            questions: 问题列表

        Returns:
            List[str]: 答案列表
        """
        if not self.chat_service:
            raise RuntimeError("答案生成器未初始化，请先调用initialize()方法")

        self.logger.info(f"开始生成 {len(questions)} 个问题的答案...")
        answers = []

        for i, question in enumerate(questions):
            self.logger.info(
                f"正在生成第 {i + 1}/{len(questions)} 个答案: {question[:50]}..."
            )

            try:
                answer = await self._generate_single_answer(question)
                answers.append(answer)
                self.logger.info(f"答案生成完成 ({i + 1}/{len(questions)})")

                # 添加短暂延迟，避免API调用过于频繁
                if i < len(questions) - 1:  # 最后一个不需要延迟
                    await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"生成答案失败 {question}: {e}", exc_info=True)
                answers.append(f"生成失败: {str(e)}")

        self.logger.info(
            f"所有答案生成完成，成功: {len([a for a in answers if not a.startswith('生成失败')])}/{len(questions)}"
        )
        return answers

    async def _generate_single_answer(self, question: str) -> str:
        """生成单个问题的答案

        Args:
            question: 问题文本

        Returns:
            str: 生成的答案
        """
        # 收集流式响应
        full_answer = ""

        try:
            # 获取知识库ID (如果使用知识库模式)
            knowledge_base_id = None
            search_k = 3
            if (
                self.config.dataset.type == "knowledge_base"
                and self.config.dataset.knowledge_base
            ):
                knowledge_base_id = self.config.dataset.knowledge_base.kb_id
                search_k = self.config.dataset.knowledge_base.search_k

            # 调用ChatSev的stream_chat方法
            async for chunk_dict in self.chat_service.stream_chat(
                question=question,
                api_key=self.config.llm_config.api_key,
                supplier=self.config.llm_config.supplier,
                model=self.config.llm_config.model,
                session_id="evaluation_session",
                knowledge_base_id=knowledge_base_id,
                search_k=search_k,
                temperature=self.config.llm_config.temperature,
            ):
                # 只收集实际的答案内容chunk
                if chunk_dict.get("type") == "chunk":
                    chunk_data = chunk_dict.get("data", "")
                    if chunk_data:
                        full_answer += chunk_data

            return full_answer.strip() if full_answer else "未能生成有效答案"

        except Exception as e:
            self.logger.error(f"调用ChatSev失败: {e}", exc_info=True)
            raise

    async def get_contexts_for_questions(self, questions: List[str]) -> List[List[str]]:
        """为问题列表获取检索上下文

        Args:
            questions: 问题列表

        Returns:
            List[List[str]]: 每个问题对应的上下文列表
        """
        if not self.knowledge:
            self.logger.warning("未初始化知识库，返回空上下文")
            return [[] for _ in questions]

        self.logger.info(f"为 {len(questions)} 个问题获取检索上下文...")
        contexts = []

        # 获取配置参数
        kb_id = self.config.dataset.knowledge_base.kb_id
        search_k = self.config.dataset.knowledge_base.search_k
        filter_dict = {}

        # 如果有文件MD5过滤
        if self.config.dataset.knowledge_base.filter_by_file_md5:
            filter_dict["source_file_md5"] = {
                "$in": [self.config.dataset.knowledge_base.filter_by_file_md5]
            }

        for i, question in enumerate(questions):
            try:
                self.logger.info(
                    f"检索第 {i + 1}/{len(questions)} 个问题的上下文: {question[:50]}..."
                )

                # 获取检索器
                retriever = await self.knowledge.get_retriever_for_knowledge_base(
                    kb_id=kb_id,
                    filter_dict=filter_dict if filter_dict else None,
                    search_k=search_k,
                )

                # 执行检索
                documents = await retriever.ainvoke(question)

                # 提取文档内容
                context_list = [
                    doc.page_content
                    for doc in documents
                    if hasattr(doc, "page_content")
                ]
                contexts.append(context_list)

                self.logger.info(f"检索到 {len(context_list)} 个上下文片段")

            except Exception as e:
                self.logger.error(f"检索上下文失败 {question}: {e}", exc_info=True)
                contexts.append([])

        self.logger.info(
            f"上下文检索完成，平均每个问题 {sum(len(c) for c in contexts) / len(contexts):.1f} 个片段"
        )
        return contexts

    async def batch_generate_answers(
        self, questions: List[str], batch_size: int = 5
    ) -> List[str]:
        """批量并发生成答案

        Args:
            questions: 问题列表
            batch_size: 并发批次大小

        Returns:
            List[str]: 答案列表
        """
        if not self.chat_service:
            raise RuntimeError("答案生成器未初始化，请先调用initialize()方法")

        self.logger.info(
            f"开始批量生成答案，总计 {len(questions)} 个问题，批次大小: {batch_size}"
        )
        all_answers = []

        # 分批处理
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            self.logger.info(
                f"处理第 {i // batch_size + 1} 批，包含 {len(batch_questions)} 个问题"
            )

            # 并发生成当前批次的答案
            tasks = [
                self._generate_single_answer(question) for question in batch_questions
            ]

            try:
                batch_answers = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理异常结果
                processed_answers = []
                for j, result in enumerate(batch_answers):
                    if isinstance(result, Exception):
                        self.logger.error(f"批次中第 {j + 1} 个问题生成失败: {result}")
                        processed_answers.append(f"生成失败: {str(result)}")
                    else:
                        processed_answers.append(result)

                all_answers.extend(processed_answers)

                # 批次间延迟
                if i + batch_size < len(questions):
                    await asyncio.sleep(1.0)

            except Exception as e:
                self.logger.error(f"批次处理失败: {e}", exc_info=True)
                # 如果批次失败，为该批次的所有问题添加错误答案
                all_answers.extend([f"批次生成失败: {str(e)}"] * len(batch_questions))

        self.logger.info(
            f"批量答案生成完成，成功: {len([a for a in all_answers if not a.startswith('生成失败') and not a.startswith('批次生成失败')])}/{len(questions)}"
        )
        return all_answers
