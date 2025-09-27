"""
主评估流程协调器
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_schema import EvaluationConfig
from config.validator import ConfigValidator

from .answer_generator import AnswerGenerator
from .data_loader import DataLoader
from .evaluator import RAGASEvaluator


class MainEvaluator:
    """主评估流程协调器"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化各组件
        self.data_loader = DataLoader(config.dataset)
        self.answer_generator = AnswerGenerator(config)
        self.evaluator = RAGASEvaluator(config)

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        validator = ConfigValidator()
        if not validator.validate(self.config):
            errors = validator.get_validation_report()["errors"]
            raise ValueError(f"配置验证失败: {errors}")

        # 记录警告
        warnings = validator.get_validation_report()["warnings"]
        for warning in warnings:
            self.logger.warning(warning)

    async def run_evaluation(self) -> Dict[str, Any]:
        """运行完整的评估流程

        Returns:
            Dict[str, Any]: 评估结果
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("开始RAG评估流程...")
        self.logger.info(f"项目: {self.config.project_name}")
        self.logger.info(f"描述: {self.config.description}")
        self.logger.info("=" * 60)

        try:
            # 1. 初始化组件
            await self._initialize_components()

            # 2. 加载数据集
            dataset = await self._load_dataset()

            # 3. 生成答案
            answers = await self._generate_answers(dataset)

            # 4. 获取上下文（如果需要）
            contexts = await self._get_contexts(dataset, answers)

            # 5. 执行评估
            evaluation_results = await self._run_evaluation(dataset, answers, contexts)

            # 6. 保存结果
            await self._save_results(evaluation_results, dataset, answers, contexts)

            # 7. 生成报告
            final_results = await self._generate_final_report(
                evaluation_results, dataset, answers, contexts, start_time
            )

            self.logger.info("=" * 60)
            self.logger.info("RAG评估流程完成！")
            self.logger.info("=" * 60)

            return final_results

        except Exception as e:
            self.logger.error(f"评估流程发生错误: {e}", exc_info=True)
            raise

    async def _initialize_components(self):
        """初始化组件"""
        self.logger.info("1/6 初始化组件...")
        await self.answer_generator.initialize()
        self.logger.info("组件初始化完成")

    async def _load_dataset(self):
        """加载数据集"""
        self.logger.info("2/6 加载数据集...")
        dataset = await self.data_loader.load_dataset()

        self.logger.info("数据集加载完成:")
        self.logger.info(f"  - 问题数量: {len(dataset.questions)}")
        self.logger.info(f"  - 真实答案数量: {len(dataset.ground_truths)}")
        self.logger.info(f"  - 预设上下文: {'是' if dataset.contexts else '否'}")

        return dataset

    async def _generate_answers(self, dataset):
        """生成答案"""
        self.logger.info("3/6 生成答案...")

        # 检查是否需要批量生成
        if len(dataset.questions) > 10:
            self.logger.info("使用批量并发模式生成答案...")
            answers = await self.answer_generator.batch_generate_answers(
                dataset.questions, batch_size=5
            )
        else:
            self.logger.info("使用串行模式生成答案...")
            answers = await self.answer_generator.generate_answers(dataset.questions)

        # 统计生成结果
        success_count = len(
            [
                a
                for a in answers
                if not a.startswith("生成失败") and not a.startswith("批次生成失败")
            ]
        )
        self.logger.info(f"答案生成完成: 成功 {success_count}/{len(answers)}")

        return answers

    async def _get_contexts(self, dataset, answers):
        """获取上下文"""
        self.logger.info("4/6 获取上下文...")

        if dataset.contexts is not None:
            self.logger.info("使用预设上下文")
            contexts = dataset.contexts
        else:
            self.logger.info("从知识库检索上下文...")
            contexts = await self.answer_generator.get_contexts_for_questions(
                dataset.questions
            )

        # 统计上下文信息
        total_contexts = sum(len(ctx) for ctx in contexts)
        avg_contexts = total_contexts / len(contexts) if contexts else 0
        self.logger.info(
            f"上下文获取完成: 总计 {total_contexts} 个片段，平均每题 {avg_contexts:.1f} 个"
        )

        return contexts

    async def _run_evaluation(self, dataset, answers, contexts):
        """执行评估"""
        self.logger.info("5/6 执行RAGAS评估...")

        self.logger.info("评估配置:")
        self.logger.info(f"  - 评估指标: {', '.join(self.config.metrics)}")
        self.logger.info(
            f"  - 评估模型: {self.config.evaluator_config.judge_llm.supplier}/{self.config.evaluator_config.judge_llm.model}"
        )

        evaluation_results = await self.evaluator.evaluate(
            questions=dataset.questions,
            answers=answers,
            contexts=contexts,
            ground_truths=dataset.ground_truths,
        )

        # 输出评估摘要
        summary = evaluation_results.get("summary", {})
        self.logger.info("评估完成:")
        self.logger.info(f"  - 平均分数: {summary.get('average_score', 0):.4f}")
        self.logger.info(f"  - 最佳指标: {summary.get('best_metric', 'N/A')}")
        self.logger.info(f"  - 最差指标: {summary.get('worst_metric', 'N/A')}")

        return evaluation_results

    async def _save_results(self, evaluation_results, dataset, answers, contexts):
        """保存结果"""
        self.logger.info("6/6 保存评估结果...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 准备完整的结果数据
        complete_results = {
            "config": {
                "project_name": self.config.project_name,
                "description": self.config.description,
                "version": self.config.version,
                "metrics": self.config.metrics,
                "dataset_type": self.config.dataset.type,
                "llm_config": {
                    "supplier": self.config.llm_config.supplier,
                    "model": self.config.llm_config.model,
                    "temperature": self.config.llm_config.temperature,
                },
            },
            "evaluation_results": evaluation_results,
            "raw_data": {
                "questions": dataset.questions,
                "answers": answers,
                "contexts": contexts,
                "ground_truths": dataset.ground_truths,
            },
            "timestamp": timestamp,
        }

        # 保存JSON格式
        if "json" in self.config.output.export_format:
            json_file = results_dir / f"evaluation_results_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(
                    complete_results, f, ensure_ascii=False, indent=2, default=str
                )
            self.logger.info(f"JSON结果已保存: {json_file}")

        # 保存CSV格式
        if "csv" in self.config.output.export_format:
            csv_file = results_dir / f"evaluation_results_{timestamp}.csv"
            if "detailed_results" in evaluation_results:
                evaluation_results["detailed_results"].to_csv(
                    csv_file, index=False, encoding="utf-8"
                )
                self.logger.info(f"CSV结果已保存: {csv_file}")

        # 保存HTML报告
        if "html" in self.config.output.export_format:
            html_file = results_dir / f"evaluation_report_{timestamp}.html"
            await self._generate_html_report(complete_results, html_file)
            self.logger.info(f"HTML报告已保存: {html_file}")

        self.logger.info(f"所有结果已保存到: {results_dir}")

    async def _generate_html_report(self, results: Dict[str, Any], output_file: Path):
        """生成HTML报告"""
        try:
            eval_results = results["evaluation_results"]
            config = results["config"]

            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG评估报告 - {config["project_name"]}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric-card {{ background-color: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        .table th {{ background-color: #f8f9fa; }}
        .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG评估报告</h1>
        <p><strong>项目:</strong> {config["project_name"]}</p>
        <p><strong>描述:</strong> {config["description"]}</p>
        <p><strong>评估时间:</strong> {results["timestamp"]}</p>
        <p><strong>数据集类型:</strong> {config["dataset_type"]}</p>
        <p><strong>LLM模型:</strong> {config["llm_config"]["supplier"]}/{config["llm_config"]["model"]}</p>
    </div>
    
    <div class="summary">
        <h2>评估摘要</h2>
        <p><strong>平均分数:</strong> <span class="score">{eval_results.get("summary", {}).get("average_score", 0):.4f}</span></p>
        <p><strong>样本数量:</strong> {eval_results.get("metadata", {}).get("total_samples", 0)}</p>
        <p><strong>评估指标:</strong> {", ".join(config["metrics"])}</p>
    </div>
    
    <h2>指标详情</h2>
"""

            # 添加各指标的详细信息
            overall_scores = eval_results.get("overall_scores", {})
            for metric, score in overall_scores.items():
                html_content += f"""
    <div class="metric-card">
        <h3>{metric}</h3>
        <div class="score">{score:.4f}</div>
    </div>
"""

            html_content += """
</body>
</html>
"""

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {e}", exc_info=True)

    async def _generate_final_report(
        self, evaluation_results, dataset, answers, contexts, start_time
    ):
        """生成最终报告"""
        end_time = datetime.now()
        duration = end_time - start_time

        final_results = {
            "project_info": {
                "name": self.config.project_name,
                "description": self.config.description,
                "version": self.config.version,
            },
            "execution_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_formatted": str(duration),
            },
            "dataset_info": {
                "type": self.config.dataset.type,
                "total_questions": len(dataset.questions),
                "has_predefined_contexts": dataset.contexts is not None,
            },
            "generation_info": {
                "successful_answers": len(
                    [a for a in answers if not a.startswith("生成失败")]
                ),
                "total_answers": len(answers),
                "success_rate": len(
                    [a for a in answers if not a.startswith("生成失败")]
                )
                / len(answers)
                if answers
                else 0,
            },
            "evaluation_results": evaluation_results,
            "performance_summary": evaluation_results.get("summary", {}),
            "overall_scores": evaluation_results.get("overall_scores", {}),
            "metadata": evaluation_results.get("metadata", {}),
        }

        return final_results

    async def run_quick_evaluation(
        self, questions: list, expected_answers: list = None
    ) -> Dict[str, Any]:
        """运行快速评估（用于小规模测试）

        Args:
            questions: 问题列表
            expected_answers: 期望答案列表（可选）

        Returns:
            Dict[str, Any]: 快速评估结果
        """
        self.logger.info(f"开始快速评估，包含 {len(questions)} 个问题")

        try:
            # 初始化
            await self.answer_generator.initialize()

            # 生成答案
            answers = await self.answer_generator.generate_answers(questions)

            # 获取上下文
            contexts = await self.answer_generator.get_contexts_for_questions(questions)

            # 如果没有提供期望答案，使用问题作为占位符
            if not expected_answers:
                expected_answers = [
                    [f"期望答案_{i + 1}"] for i in range(len(questions))
                ]

            # 执行评估
            results = await self.evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=expected_answers,
            )

            self.logger.info("快速评估完成")
            return results

        except Exception as e:
            self.logger.error(f"快速评估失败: {e}", exc_info=True)
            raise
