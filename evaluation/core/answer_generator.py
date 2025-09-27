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

    def __init__(self, config: EvaluationConfig, max_concurrent: int = 10):
        self.config = config
        self.knowledge: Optional[Knowledge] = None
        self.chat_service: Optional[ChatSev] = None
        self.logger = logging.getLogger(__name__)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

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

    async def generate_answers(
        self, questions: List[str]
    ) -> tuple[List[str], List[List[str]]]:
        """为问题列表生成答案，并收集使用的上下文（异步并发版本）

        Args:
            questions: 问题列表

        Returns:
            tuple[List[str], List[List[str]]]: (答案列表, 上下文列表)
        """
        if not self.chat_service:
            raise RuntimeError("答案生成器未初始化，请先调用initialize()方法")

        self.logger.info(
            f"开始并发生成 {len(questions)} 个问题的答案，最大并发数: {self.max_concurrent}"
        )

        # 创建所有任务
        tasks = []
        for i, question in enumerate(questions):
            task = self._generate_single_answer_with_semaphore(
                question, i + 1, len(questions)
            )
            tasks.append(task)

        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        answers = []
        all_contexts = []
        success_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"生成第 {i + 1} 个答案失败: {result}", exc_info=True)
                answers.append(f"生成失败: {str(result)}")
                all_contexts.append([])
            elif isinstance(result, tuple) and len(result) == 2:
                answer, contexts = result
                answers.append(answer)
                all_contexts.append(contexts)
                if not answer.startswith("生成失败"):
                    success_count += 1
            else:
                self.logger.error(f"第 {i + 1} 个答案返回格式异常: {result}")
                answers.append("返回格式异常")
                all_contexts.append([])

        self.logger.info(f"并发答案生成完成，成功: {success_count}/{len(questions)}")
        return answers, all_contexts

    async def _generate_single_answer_with_semaphore(
        self, question: str, index: int, total: int
    ) -> tuple[str, list[str]]:
        """带信号量控制的单个答案生成方法

        Args:
            question: 问题文本
            index: 当前问题索引（从1开始）
            total: 总问题数

        Returns:
            tuple[str, list[str]]: (生成的答案, 使用的上下文列表)
        """
        async with self.semaphore:
            self.logger.info(f"开始生成第 {index}/{total} 个答案: {question[:50]}...")

            try:
                answer, contexts = await self._generate_single_answer(question)
                self.logger.info(
                    f"答案生成完成 ({index}/{total})，获得 {len(contexts)} 个上下文"
                )
                return answer, contexts
            except Exception as e:
                self.logger.error(f"生成第 {index} 个答案失败: {e}", exc_info=True)
                return f"生成失败: {str(e)}", []

    async def _generate_single_answer(self, question: str) -> tuple[str, list[str]]:
        """生成单个问题的答案，并收集使用的上下文

        Args:
            question: 问题文本

        Returns:
            tuple[str, list[str]]: (生成的答案, 使用的上下文列表)
        """
        # 收集流式响应
        full_answer = ""
        contexts = []

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
                # 收集答案内容chunk
                if chunk_dict.get("type") == "chunk":
                    chunk_data = chunk_dict.get("data", "")
                    if chunk_data:
                        full_answer += chunk_data

                # 收集上下文信息（从tool_result中提取）
                elif chunk_dict.get("type") == "tool_result":
                    tool_data = chunk_dict.get("data", {})
                    if tool_data.get("name") == "知识库检索":
                        context_json = tool_data.get("content", "")
                        if context_json:
                            try:
                                import json

                                context_docs = json.loads(context_json)
                                for doc in context_docs:
                                    if isinstance(doc, dict) and "page_content" in doc:
                                        contexts.append(doc["page_content"])
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"解析上下文JSON失败: {e}")

            answer = full_answer.strip() if full_answer else "未能生成有效答案"
            return answer, contexts

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
    ) -> tuple[List[str], List[List[str]]]:
        """批量并发生成答案（已废弃，建议使用 generate_answers 方法）

        Args:
            questions: 问题列表
            batch_size: 并发批次大小

        Returns:
            tuple[List[str], List[List[str]]]: (答案列表, 上下文列表)
        """
        if not self.chat_service:
            raise RuntimeError("答案生成器未初始化，请先调用initialize()方法")

        self.logger.warning(
            "batch_generate_answers 方法已废弃，将使用 generate_answers 方法"
        )
        return await self.generate_answers(questions)
