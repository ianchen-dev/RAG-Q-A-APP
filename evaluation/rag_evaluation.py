"""
RAG评估系统主执行脚本
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """设置环境变量"""
    print(f"项目根目录: {project_root}")

    # 检查 .env 文件
    env_file = project_root / ".env"
    print(f"检查 .env 文件: {env_file}, 存在: {env_file.exists()}")

    if env_file.exists():
        load_dotenv(dotenv_path=env_file)  # 加载 .env 基础配置
        print("已加载 .env 基础配置")

    # 尝试加载开发环境配置
    dev_env_file = project_root / ".env.dev"
    print(f"检查 .env.dev 文件: {dev_env_file}, 存在: {dev_env_file.exists()}")

    if dev_env_file.exists():
        print("加载开发环境配置 .env.dev")
        load_dotenv(dotenv_path=dev_env_file, override=True)
        print("已加载 .env.dev 配置")


setup_environment()

from core.main_evaluator import MainEvaluator

from config.config_schema import load_config
from src.config.database_manager import close_databases, get_database_manager
from src.config.logging_config import setup_development_logging


def setup_logging(verbose: bool = False) -> logging.Logger:
    """设置日志系统"""
    log_level = logging.DEBUG if verbose else logging.INFO

    # 使用项目的日志配置
    logger = setup_development_logging(log_dir="./evaluation/logs")

    # 设置当前模块的日志级别
    logging.getLogger().setLevel(log_level)

    return logger


async def initialize_databases(logger: logging.Logger):
    """初始化数据库连接和 Beanie ODM"""
    try:
        logger.info("正在初始化数据库连接...")

        # 初始化数据库管理器，这会自动初始化 Beanie ODM
        db_manager = await get_database_manager()

        # 验证连接
        await db_manager.health_check(force=True)
        logger.info("数据库连接初始化成功")

        return db_manager
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}", exc_info=True)
        raise


async def main():
    """主函数"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="RAG系统评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python evaluation/rag_evaluation.py --config evaluation/examples/sample_config.yaml
  python evaluation/rag_evaluation.py --config my_config.yaml --verbose
  python evaluation/rag_evaluation.py --quick-test
        """,
    )

    parser.add_argument("--config", type=str, help="配置文件路径 (YAML格式)")

    parser.add_argument("--verbose", action="store_true", help="详细日志输出")

    parser.add_argument(
        "--quick-test", action="store_true", help="运行快速测试（使用内置示例数据）"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="仅验证配置，不执行实际评估"
    )

    parser.add_argument(
        "--concurrent", type=int, default=10, help="最大并发数（默认10）"
    )

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        logger.info("=" * 60)
        logger.info("RAG评估系统启动")
        logger.info("=" * 60)

        # 初始化数据库连接
        await initialize_databases(logger)

        if args.quick_test:
            await run_quick_test(logger, args.concurrent)
        elif args.config:
            await run_evaluation_from_config(
                args.config, logger, args.dry_run, args.concurrent
            )
        else:
            logger.error("请指定配置文件 (--config) 或使用快速测试 (--quick-test)")
            parser.print_help()
            return 1

        logger.info("=" * 60)
        logger.info("RAG评估系统执行完成")
        logger.info("=" * 60)
        return 0

    except KeyboardInterrupt:
        logger.info("用户中断执行")
        return 1
    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}", exc_info=True)
        return 1
    finally:
        # 清理数据库连接
        try:
            await close_databases()
            logger.info("数据库连接已清理")
        except Exception as e:
            logger.warning(f"清理数据库连接时出现警告: {e}")


async def run_evaluation_from_config(
    config_path: str,
    logger: logging.Logger,
    dry_run: bool = False,
    max_concurrent: int = 10,
):
    """从配置文件运行评估"""
    logger.info(f"从配置文件运行评估: {config_path}")

    # 检查配置文件是否存在
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        # 加载配置
        logger.info("加载配置文件...")
        config = load_config(config_path)
        logger.info("配置加载成功:")
        logger.info(f"  - 项目: {config.project_name}")
        logger.info(f"  - 描述: {config.description}")
        logger.info(f"  - 数据集类型: {config.dataset.type}")
        logger.info(f"  - 评估指标: {', '.join(config.metrics)}")

        if dry_run:
            logger.info("干运行模式：配置验证完成，跳过实际评估")
            return

        # 创建评估器
        logger.info(f"创建评估器，最大并发数: {max_concurrent}")
        evaluator = MainEvaluator(config, max_concurrent=max_concurrent)

        # 运行评估
        logger.info("开始执行评估...")
        results = await evaluator.run_evaluation()

        # 输出摘要
        print_evaluation_summary(results, logger)

    except Exception as e:
        logger.error(f"配置文件评估失败: {e}", exc_info=True)
        raise


async def run_quick_test(logger: logging.Logger, max_concurrent: int = 10):
    """运行快速测试"""
    logger.info("运行快速测试...")

    # 创建测试配置
    test_config = create_test_config()

    try:
        # 创建评估器
        logger.info(f"创建快速测试评估器，最大并发数: {max_concurrent}")
        evaluator = MainEvaluator(test_config, max_concurrent=max_concurrent)

        # 运行快速评估
        test_questions = [
            "什么是检索增强生成（RAG）？",
            "RAGAS评估框架的作用是什么？",
            "如何提高RAG系统的准确性？",
        ]

        test_expected_answers = [
            ["RAG是一种结合检索和生成的AI技术，通过检索相关文档来增强答案生成。"],
            ["RAGAS是专门用于评估RAG系统性能的评估框架，提供多种评估指标。"],
            ["可以通过优化检索质量、改进重排序、调整提示词等方式提高RAG系统准确性。"],
        ]

        results = await evaluator.run_quick_evaluation(
            test_questions, test_expected_answers
        )

        logger.info("快速测试完成")
        print_evaluation_summary({"evaluation_results": results}, logger)

    except Exception as e:
        logger.error(f"快速测试失败: {e}", exc_info=True)
        raise


def create_test_config():
    """创建测试配置"""
    from config.config_schema import (
        DatasetConfig,
        EvaluationConfig,
        EvaluatorConfig,
        JudgeLLMConfig,
        LLMConfig,
        OutputConfig,
    )

    # 使用文件模式的测试配置
    dataset_config = DatasetConfig(
        type="file",
        questions_path="evaluation/examples/sample_questions.json",
        ground_truths_path="evaluation/examples/sample_ground_truths.json",
    )

    llm_config = LLMConfig(
        supplier="siliconflow",  # 使用默认的供应商
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.1,
    )

    judge_llm_config = JudgeLLMConfig(
        supplier="siliconflow", model="deepseek-ai/DeepSeek-V3", temperature=0.0
    )

    evaluator_config = EvaluatorConfig(judge_llm=judge_llm_config)

    output_config = OutputConfig(
        results_dir="evaluation/results/quick_test",
        export_format=["json", "csv"],
        save_individual_scores=True,
        generate_charts=False,
    )

    return EvaluationConfig(
        project_name="RAG评估系统快速测试",
        description="使用内置示例数据进行快速功能测试",
        version="1.0.0",
        dataset=dataset_config,
        llm_config=llm_config,
        metrics=["AnswerRelevancy", "Faithfulness"],  # 简化指标以加快测试
        evaluator_config=evaluator_config,
        output=output_config,
    )


def print_evaluation_summary(results: dict, logger: logging.Logger):
    """打印评估摘要"""
    eval_results = results.get("evaluation_results", results)
    overall_scores = eval_results.get("overall_scores", {})
    summary = eval_results.get("summary", {})

    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)

    if "project_info" in results:
        print(f"项目: {results['project_info']['name']}")
        print(f"描述: {results['project_info']['description']}")
        print("-" * 60)

    if overall_scores:
        print("各指标分数:")
        for metric, score in overall_scores.items():
            print(f"  {metric:20s}: {score:.4f}")
        print("-" * 60)

    if summary:
        print(f"平均分数: {summary.get('average_score', 0):.4f}")

        best_metric = summary.get("best_metric")
        if best_metric:
            print(f"最佳指标: {best_metric[0]} ({best_metric[1]:.4f})")

        worst_metric = summary.get("worst_metric")
        if worst_metric:
            print(f"最差指标: {worst_metric[0]} ({worst_metric[1]:.4f})")

    if "execution_info" in results:
        exec_info = results["execution_info"]
        print(f"执行时间: {exec_info.get('duration_formatted', 'N/A')}")

    if "generation_info" in results:
        gen_info = results["generation_info"]
        success_rate = gen_info.get("success_rate", 0) * 100
        print(
            f"答案生成成功率: {success_rate:.1f}% ({gen_info.get('successful_answers', 0)}/{gen_info.get('total_answers', 0)})"
        )

    print("=" * 60)


if __name__ == "__main__":
    exit(asyncio.run(main()))
