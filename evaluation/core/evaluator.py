"""
RAGAS评估器模块
基于RAGAS 0.3.5+实现RAG系统评估
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,  # 答案相关性
        ContextPrecision,  # 上下文精确度
        ContextRecall,  # 上下文召回率
        ContextRelevance,  # 上下文相关性
        Faithfulness,  # 忠实度
    )

    RAGAS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RAGAS模块导入失败: {e}")
    RAGAS_AVAILABLE = False

from config.config_schema import EvaluationConfig
from src.utils.llm_modle import get_llms


class RAGASEvaluator:
    """RAGAS评估器，使用0.3.5+版本API"""

    # 指标映射
    METRICS_MAP = {}

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 检查RAGAS是否可用
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS模块不可用。请安装RAGAS: pip install ragas>=0.3.5")

        # 初始化指标映射
        self._initialize_metrics()

        # 选择要使用的指标
        self.selected_metrics = []
        for metric_name in config.metrics:
            if metric_name in self.METRICS_MAP:
                self.selected_metrics.append(self.METRICS_MAP[metric_name])
            else:
                self.logger.warning(f"未知的评估指标: {metric_name}")

        if not self.selected_metrics:
            raise ValueError("没有可用的评估指标")

        self.logger.info(f"初始化评估器，使用指标: {config.metrics}")

    def _initialize_metrics(self):
        """初始化RAGAS指标"""
        try:
            # 先尝试不使用LLM参数初始化（避免版本兼容性问题）
            self.METRICS_MAP = {
                "ContextRelevance": ContextRelevance(),
                "ContextPrecision": ContextPrecision(),
                "ContextRecall": ContextRecall(),
                "Faithfulness": Faithfulness(),
                "AnswerRelevancy": AnswerRelevancy(),  # 使用新的类名
            }
            self.logger.info("使用默认配置初始化RAGAS指标")

        except Exception as e:
            self.logger.error(f"初始化RAGAS指标失败: {e}", exc_info=True)
            # 如果默认初始化失败，尝试使用LLM参数
            try:
                judge_llm = self._get_judge_llm()
                if judge_llm:
                    self.METRICS_MAP = {
                        "ContextRelevance": ContextRelevance(llm=judge_llm),
                        "ContextPrecision": ContextPrecision(llm=judge_llm),
                        "ContextRecall": ContextRecall(llm=judge_llm),
                        "Faithfulness": Faithfulness(llm=judge_llm),
                        "AnswerRelevancy": AnswerRelevancy(llm=judge_llm),
                    }
                    self.logger.info("使用自定义LLM初始化RAGAS指标")
                else:
                    raise Exception("无法获取评估LLM")
            except Exception as e2:
                self.logger.error(f"LLM初始化也失败: {e2}")
                raise

    def _get_judge_llm(self):
        """获取评估用的LLM"""
        try:
            judge_config = self.config.evaluator_config.judge_llm
            llm = get_llms(
                supplier=judge_config.supplier,
                model=judge_config.model,
                api_key=judge_config.api_key,
                temperature=judge_config.temperature,
            )

            # 为LLM添加速率限制配置
            if hasattr(llm, "request_timeout"):
                llm.request_timeout = 60  # 设置请求超时
            if hasattr(llm, "max_retries"):
                llm.max_retries = 3  # 设置最大重试次数

            return llm
        except Exception as e:
            self.logger.warning(f"获取评估LLM失败: {e}")
            return None

    def _get_embeddings(self):
        """获取嵌入模型"""
        try:
            from src.utils.embedding import get_embedding

            judge_config = self.config.evaluator_config.judge_embedding
            # 使用项目中已有的嵌入模型获取函数
            embeddings = get_embedding(judge_config.supplier, judge_config.model)
            return embeddings
        except Exception as e:
            self.logger.error(f"获取嵌入模型失败: {e}")

    async def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[List[str]],
    ) -> Dict[str, Any]:
        """执行RAGAS评估

        Args:
            questions: 问题列表
            answers: 答案列表
            contexts: 上下文列表
            ground_truths: 真实答案列表

        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info("开始执行RAGAS评估...")

        # 验证输入数据
        self._validate_input_data(questions, answers, contexts, ground_truths)

        # 预处理数据
        processed_data = self._preprocess_data(
            questions, answers, contexts, ground_truths
        )

        # 构建RAGAS数据集
        dataset_dict = {
            "question": processed_data["questions"],
            "answer": processed_data["answers"],
            "contexts": processed_data["contexts"],
            "ground_truth": processed_data["ground_truths"],
        }

        dataset = Dataset.from_dict(dataset_dict)
        self.logger.info(f"构建数据集完成，包含 {len(dataset)} 条记录")

        # 执行评估
        try:
            self.logger.info("开始RAGAS评估计算...")

            # 在异步环境中运行同步的evaluate函数
            # 获取评估用的LLM，确保RAGAS使用我们配置的LLM而不是默认的OpenAI
            judge_llm = self._get_judge_llm()
            self.logger.info(f"获取到的评估LLM: {type(judge_llm)} - {judge_llm}")

            if judge_llm is None:
                self.logger.error("评估LLM为None，无法进行RAGAS评估")
                raise RuntimeError("评估LLM配置失败")

            # 获取嵌入模型，避免RAGAS创建默认的OpenAI嵌入模型
            embeddings = self._get_embeddings()
            self.logger.info(f"获取到的嵌入模型: {type(embeddings)} - {embeddings}")

            if embeddings is None:
                self.logger.error("嵌入模型为None，无法进行RAGAS评估")
                raise RuntimeError("嵌入模型配置失败")

            # 设置评估配置，降低并发以避免API限制
            from ragas.run_config import RunConfig

            run_config = RunConfig(
                max_workers=2,  # 进一步降低并发，避免超时
                max_wait=600,  # 增加最大等待时间
                timeout=120,  # 增加单个请求超时时间
            )

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: evaluate(
                    dataset=dataset,
                    metrics=self.selected_metrics,
                    llm=judge_llm,  # 明确传递LLM，避免RAGAS创建默认OpenAI客户端
                    embeddings=embeddings,  # 明确传递嵌入模型，避免RAGAS创建默认OpenAI嵌入
                    run_config=run_config,  # 添加运行配置控制并发
                ),
            )

            self.logger.info("RAGAS评估计算完成")
            self.logger.info(f"RAGAS原始结果类型: {type(result)}")
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                self.logger.info(f"RAGAS结果列名: {list(df.columns)}")
                self.logger.info(f"RAGAS结果前几行:\n{df.head()}")

                # 详细分析每列的数据状态
                for col in df.columns:
                    if col in [
                        "nv_context_relevance",
                        "context_precision",
                        "context_recall",
                        "faithfulness",
                        "answer_relevancy",
                    ]:
                        null_count = df[col].isna().sum()
                        total_count = len(df[col])
                        valid_count = total_count - null_count
                        if valid_count > 0:
                            self.logger.info(
                                f"列 {col}: {valid_count}/{total_count} 有效值, 平均值: {df[col].dropna().mean():.4f}"
                            )
                        else:
                            self.logger.warning(
                                f"列 {col}: {valid_count}/{total_count} 有效值 (全部为NaN)"
                            )
            else:
                self.logger.info(f"RAGAS结果内容: {result}")

            return self._process_results(result, processed_data)

        except Exception as e:
            self.logger.error(f"RAGAS评估失败: {e}", exc_info=True)
            # 返回一个备用的结果结构
            return self._create_fallback_results(
                questions, answers, contexts, ground_truths
            )

    def _validate_input_data(self, questions, answers, contexts, ground_truths):
        """验证输入数据的一致性"""
        lengths = [len(questions), len(answers), len(contexts), len(ground_truths)]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"输入数据长度不一致: questions={len(questions)}, answers={len(answers)}, contexts={len(contexts)}, ground_truths={len(ground_truths)}"
            )

        if lengths[0] == 0:
            raise ValueError("输入数据为空")

        self.logger.info(f"输入数据验证通过，包含 {lengths[0]} 条记录")

    def _preprocess_data(self, questions, answers, contexts, ground_truths):
        """预处理数据，确保格式正确"""
        processed_questions = []
        processed_answers = []
        processed_contexts = []
        processed_ground_truths = []

        for i, (q, a, c, gt) in enumerate(
            zip(questions, answers, contexts, ground_truths)
        ):
            try:
                # 处理问题 - 确保是字符串
                processed_q = str(q).strip() if q else f"问题{i + 1}"

                # 处理答案 - 确保是字符串
                processed_a = str(a).strip() if a else "无答案"

                # 处理上下文 - 确保是字符串列表
                if isinstance(c, list):
                    processed_c = [str(item).strip() for item in c if item]
                elif isinstance(c, str):
                    processed_c = [c.strip()] if c.strip() else []
                else:
                    processed_c = []

                # 处理真实答案 - 确保是字符串列表，RAGAS期望的是单个字符串
                if isinstance(gt, list):
                    # 将列表中的第一个作为主要答案，或者合并多个答案
                    if gt:
                        processed_gt = str(gt[0]).strip()
                    else:
                        processed_gt = "无真实答案"
                elif isinstance(gt, str):
                    processed_gt = gt.strip() if gt.strip() else "无真实答案"
                else:
                    processed_gt = "无真实答案"

                processed_questions.append(processed_q)
                processed_answers.append(processed_a)
                processed_contexts.append(processed_c)
                processed_ground_truths.append(processed_gt)

            except Exception as e:
                self.logger.warning(f"处理第 {i + 1} 条数据时出错: {e}")
                # 使用默认值
                processed_questions.append(f"问题{i + 1}")
                processed_answers.append("处理失败")
                processed_contexts.append([])
                processed_ground_truths.append("无真实答案")

        return {
            "questions": processed_questions,
            "answers": processed_answers,
            "contexts": processed_contexts,
            "ground_truths": processed_ground_truths,
        }

    def _process_results(self, ragas_result, processed_data) -> Dict[str, Any]:
        """处理RAGAS评估结果"""
        try:
            # 提取总体分数
            overall_scores = {}

            # RAGAS结果可能是字典形式
            if hasattr(ragas_result, "to_pandas"):
                df = ragas_result.to_pandas()
            else:
                df = pd.DataFrame(ragas_result)

            # 计算各指标的平均分数
            self.logger.info(f"配置的指标: {self.config.metrics}")
            self.logger.info(f"DataFrame可用列: {list(df.columns)}")

            # RAGAS指标名称映射
            metric_mapping = {
                "ContextRelevance": "nv_context_relevance",  # 实际RAGAS返回的列名
                "ContextPrecision": "context_precision",
                "ContextRecall": "context_recall",
                "Faithfulness": "faithfulness",
                "AnswerRelevancy": "answer_relevancy",
            }

            for metric_name in self.config.metrics:
                # 首先尝试映射的名称
                mapped_name = metric_mapping.get(metric_name)
                if mapped_name and mapped_name in df.columns:
                    series = df[mapped_name]
                    # 过滤NaN值再计算平均值
                    valid_values = series.dropna()
                    if len(valid_values) > 0:
                        score = valid_values.mean()
                        self.logger.info(
                            f"指标 {metric_name} (映射为 {mapped_name}): {score} (有效值: {len(valid_values)}/{len(series)})"
                        )
                    else:
                        score = 0.0
                        self.logger.warning(
                            f"指标 {metric_name} (映射为 {mapped_name}): 所有值都是NaN，设为0"
                        )
                    overall_scores[metric_name] = score
                # 然后尝试小写名称
                elif metric_name.lower() in df.columns:
                    series = df[metric_name.lower()]
                    valid_values = series.dropna()
                    if len(valid_values) > 0:
                        score = valid_values.mean()
                        self.logger.info(
                            f"指标 {metric_name} (小写): {score} (有效值: {len(valid_values)}/{len(series)})"
                        )
                    else:
                        score = 0.0
                        self.logger.warning(
                            f"指标 {metric_name} (小写): 所有值都是NaN，设为0"
                        )
                    overall_scores[metric_name] = score
                # 最后尝试原名称
                elif metric_name in df.columns:
                    series = df[metric_name]
                    valid_values = series.dropna()
                    if len(valid_values) > 0:
                        score = valid_values.mean()
                        self.logger.info(
                            f"指标 {metric_name} (原名): {score} (有效值: {len(valid_values)}/{len(series)})"
                        )
                    else:
                        score = 0.0
                        self.logger.warning(
                            f"指标 {metric_name} (原名): 所有值都是NaN，设为0"
                        )
                    overall_scores[metric_name] = score
                else:
                    self.logger.warning(f"指标 {metric_name} 在RAGAS结果中未找到")

            # 生成摘要
            summary = self._generate_summary(overall_scores)

            return {
                "overall_scores": overall_scores,
                "detailed_results": df,
                "summary": summary,
                "metadata": {
                    "total_samples": len(processed_data["questions"]),
                    "metrics_used": self.config.metrics,
                    "evaluation_config": {
                        "judge_llm": {
                            "supplier": self.config.evaluator_config.judge_llm.supplier,
                            "model": self.config.evaluator_config.judge_llm.model,
                        }
                    },
                },
            }

        except Exception as e:
            self.logger.error(f"处理RAGAS结果失败: {e}", exc_info=True)
            return self._create_fallback_results(
                processed_data["questions"],
                processed_data["answers"],
                processed_data["contexts"],
                processed_data["ground_truths"],
            )

    def _generate_summary(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """生成评估摘要"""
        if not scores:
            return {
                "average_score": 0.0,
                "best_metric": None,
                "worst_metric": None,
                "total_metrics": 0,
            }

        valid_scores = {
            k: v
            for k, v in scores.items()
            if isinstance(v, (int, float)) and not pd.isna(v)
        }

        if not valid_scores:
            return {
                "average_score": 0.0,
                "best_metric": None,
                "worst_metric": None,
                "total_metrics": len(scores),
            }

        return {
            "average_score": sum(valid_scores.values()) / len(valid_scores),
            "best_metric": max(valid_scores.items(), key=lambda x: x[1]),
            "worst_metric": min(valid_scores.items(), key=lambda x: x[1]),
            "total_metrics": len(valid_scores),
            "score_distribution": {
                "min": min(valid_scores.values()),
                "max": max(valid_scores.values()),
                "std": pd.Series(list(valid_scores.values())).std(),
            },
        }

    def _create_fallback_results(
        self, questions, answers, contexts, ground_truths
    ) -> Dict[str, Any]:
        """创建备用结果（当RAGAS评估失败时）"""
        self.logger.warning("创建备用评估结果")

        # 简单的启发式评估
        fallback_scores = {}
        n_samples = len(questions)

        for metric_name in self.config.metrics:
            # 基于一些简单规则给出分数
            if metric_name == "AnswerRelevancy":
                # 基于答案长度的简单评分
                scores = [min(1.0, len(ans.split()) / 20) for ans in answers]
                fallback_scores[metric_name] = sum(scores) / len(scores)
            elif metric_name == "ContextRelevance":
                # 基于上下文数量的简单评分
                scores = [min(1.0, len(ctx) / 3) for ctx in contexts]
                fallback_scores[metric_name] = (
                    sum(scores) / len(scores) if scores else 0.5
                )
            else:
                # 其他指标给予默认分数
                fallback_scores[metric_name] = 0.5

        # 创建详细结果DataFrame
        detailed_data = {
            "question": questions,
            "answer": answers,
            "contexts": [str(ctx) for ctx in contexts],
            "ground_truth": [str(gt) for gt in ground_truths],
        }

        # 添加各指标的分数列
        for metric_name, score in fallback_scores.items():
            detailed_data[metric_name.lower()] = [score] * n_samples

        df = pd.DataFrame(detailed_data)

        return {
            "overall_scores": fallback_scores,
            "detailed_results": df,
            "summary": self._generate_summary(fallback_scores),
            "metadata": {
                "total_samples": n_samples,
                "metrics_used": self.config.metrics,
                "is_fallback": True,
                "warning": "使用备用评估方法，结果仅供参考",
            },
        }

    def get_supported_metrics(self) -> List[str]:
        """获取支持的评估指标列表"""
        return list(self.METRICS_MAP.keys())

    async def evaluate_single_sample(
        self, question: str, answer: str, contexts: List[str], ground_truth: str
    ) -> Dict[str, float]:
        """评估单个样本

        Args:
            question: 问题
            answer: 答案
            contexts: 上下文列表
            ground_truth: 真实答案

        Returns:
            Dict[str, float]: 各指标的分数
        """
        result = await self.evaluate([question], [answer], [contexts], [ground_truth])
        return result["overall_scores"]
