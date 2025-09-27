"""
配置验证器
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .config_schema import EvaluationConfig


class ConfigValidator:
    """配置验证器"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: EvaluationConfig) -> bool:
        """验证配置

        Args:
            config: 评估配置

        Returns:
            bool: 验证是否通过
        """
        self.errors.clear()
        self.warnings.clear()

        self._validate_dataset_config(config.dataset)
        self._validate_llm_config(config.llm_config)
        self._validate_knowledge_config(config.knowledge_config)
        self._validate_metrics(config.metrics)
        self._validate_evaluator_config(config.evaluator_config)
        self._validate_output_config(config.output)

        # 记录验证结果
        if self.errors:
            for error in self.errors:
                logging.error(f"配置验证错误: {error}")

        if self.warnings:
            for warning in self.warnings:
                logging.warning(f"配置验证警告: {warning}")

        return len(self.errors) == 0

    def _validate_dataset_config(self, dataset_config):
        """验证数据集配置"""
        if dataset_config.type == "file":
            # 文件模式验证
            if not dataset_config.questions_path:
                self.errors.append("文件模式下必须指定 questions_path")
            else:
                if not Path(dataset_config.questions_path).exists():
                    self.errors.append(
                        f"问题文件不存在: {dataset_config.questions_path}"
                    )

            if not dataset_config.ground_truths_path:
                self.errors.append("文件模式下必须指定 ground_truths_path")
            else:
                if not Path(dataset_config.ground_truths_path).exists():
                    self.errors.append(
                        f"真实答案文件不存在: {dataset_config.ground_truths_path}"
                    )

            if dataset_config.contexts_path:
                if not Path(dataset_config.contexts_path).exists():
                    self.warnings.append(
                        f"上下文文件不存在: {dataset_config.contexts_path}"
                    )

        elif dataset_config.type == "knowledge_base":
            # 知识库模式验证
            if not dataset_config.knowledge_base:
                self.errors.append("知识库模式下必须指定 knowledge_base 配置")
            else:
                kb_config = dataset_config.knowledge_base
                if not kb_config.kb_id:
                    self.errors.append("知识库模式下必须指定 kb_id")
                if kb_config.search_k <= 0:
                    self.errors.append("search_k 必须大于0")

            # 知识库模式下仍需要文件路径来加载问题和真实答案
            if not dataset_config.questions_path:
                self.errors.append(
                    "知识库模式下仍需要指定 questions_path 来加载评估问题"
                )
            else:
                if not Path(dataset_config.questions_path).exists():
                    self.errors.append(
                        f"问题文件不存在: {dataset_config.questions_path}"
                    )

            if not dataset_config.ground_truths_path:
                self.errors.append(
                    "知识库模式下仍需要指定 ground_truths_path 来加载真实答案"
                )
            else:
                if not Path(dataset_config.ground_truths_path).exists():
                    self.errors.append(
                        f"真实答案文件不存在: {dataset_config.ground_truths_path}"
                    )
        else:
            self.errors.append(f"不支持的数据集类型: {dataset_config.type}")

    def _validate_llm_config(self, llm_config):
        """验证LLM配置"""
        if not llm_config.supplier:
            self.errors.append("必须指定 LLM supplier")

        if not llm_config.model:
            self.errors.append("必须指定 LLM model")

        if llm_config.temperature < 0 or llm_config.temperature > 2:
            self.warnings.append(
                f"LLM temperature 建议在0-2之间，当前值: {llm_config.temperature}"
            )

        if llm_config.max_length <= 0:
            self.errors.append("max_length 必须大于0")

    def _validate_knowledge_config(self, knowledge_config):
        """验证知识检索配置"""
        if not knowledge_config:
            return

        if knowledge_config.bm25_k <= 0:
            self.errors.append("bm25_k 必须大于0")

        if knowledge_config.reranker_config:
            reranker = knowledge_config.reranker_config
            if reranker.use_reranker:
                if reranker.reranker_type == "remote":
                    if not reranker.remote_rerank_config:
                        self.errors.append(
                            "使用远程重排序时必须配置 remote_rerank_config"
                        )
                    else:
                        if not reranker.remote_rerank_config.api_key:
                            self.errors.append("远程重排序必须指定 api_key")
                        if not reranker.remote_rerank_config.model:
                            self.errors.append("远程重排序必须指定 model")

                if reranker.rerank_top_n <= 0:
                    self.errors.append("rerank_top_n 必须大于0")

    def _validate_metrics(self, metrics: List[str]):
        """验证评估指标"""
        if not metrics:
            self.errors.append("必须指定至少一个评估指标")

        valid_metrics = {
            "ContextRelevance",
            "ContextPrecision",
            "ContextRecall",
            "Faithfulness",
            "AnswerRelevancy",
        }

        for metric in metrics:
            if metric not in valid_metrics:
                self.errors.append(f"不支持的评估指标: {metric}")

    def _validate_evaluator_config(self, evaluator_config):
        """验证评估器配置"""
        judge_llm = evaluator_config.judge_llm

        if not judge_llm.supplier:
            self.errors.append("评估LLM必须指定 supplier")

        if not judge_llm.model:
            self.errors.append("评估LLM必须指定 model")

        if judge_llm.temperature < 0 or judge_llm.temperature > 1:
            self.warnings.append(
                f"评估LLM temperature 建议在0-1之间，当前值: {judge_llm.temperature}"
            )

    def _validate_output_config(self, output_config):
        """验证输出配置"""
        if not output_config.results_dir:
            self.errors.append("必须指定 results_dir")

        valid_formats = {"json", "csv", "html"}
        for format_type in output_config.export_format:
            if format_type not in valid_formats:
                self.errors.append(f"不支持的导出格式: {format_type}")

    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "is_valid": len(self.errors) == 0,
        }
