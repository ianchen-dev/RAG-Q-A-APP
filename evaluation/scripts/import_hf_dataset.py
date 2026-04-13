#!/usr/bin/env python3
"""
Hugging Face数据集导入脚本
用于将neural-bridge/rag-dataset-1200数据集导入到我们的评估系统中
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("import_dataset.log", encoding="utf-8"),
        ],
    )


def download_dataset() -> pd.DataFrame:
    """
    从Hugging Face下载数据集

    Returns:
        pd.DataFrame: 下载的数据集
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("请安装datasets库: pip install datasets")
        raise ImportError("Missing datasets library")

    logging.info("正在从Hugging Face下载数据集...")

    # 下载数据集
    dataset = load_dataset("neural-bridge/rag-dataset-1200")

    # 合并训练集和测试集
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    logging.info(f"数据集下载完成，共 {len(full_df)} 条记录")
    logging.info(f"训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

    return full_df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    验证数据集格式

    Args:
        df: 数据集DataFrame

    Returns:
        bool: 验证是否通过
    """
    required_columns = ["context", "question", "answer"]

    # 检查必需列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"数据集缺少必需的列: {missing_columns}")
        return False

    # 检查空值
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logging.warning(f"发现空值: {null_counts.to_dict()}")

    # 检查数据类型
    for col in required_columns:
        if not df[col].dtype == "object":
            logging.warning(f"列 {col} 的数据类型不是字符串: {df[col].dtype}")

    logging.info("数据集验证通过")
    return True


def split_dataset(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    """
    将数据集拆分成三个文件

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录

    Returns:
        Dict[str, str]: 生成的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取数据
    questions = df["question"].tolist()
    contexts = df["context"].tolist()
    answers = df["answer"].tolist()

    # 创建文件路径
    files = {
        "questions": output_dir / "rag-dataset-1200_questions.json",
        "contexts": output_dir / "rag-dataset-1200_contexts.json",
        "ground_truths": output_dir / "rag-dataset-1200_ground_truths.json",
    }

    # 保存questions文件
    questions_data = {"questions": questions}
    with open(files["questions"], "w", encoding="utf-8") as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)
    logging.info(f"问题文件保存到: {files['questions']}")

    # 保存contexts文件 (每个context作为单独的上下文列表)
    contexts_data = {"contexts": [[context] for context in contexts]}
    with open(files["contexts"], "w", encoding="utf-8") as f:
        json.dump(contexts_data, f, ensure_ascii=False, indent=2)
    logging.info(f"上下文文件保存到: {files['contexts']}")

    # 保存ground_truths文件 (每个answer作为单独的真实答案列表)
    ground_truths_data = {"ground_truths": [[answer] for answer in answers]}
    with open(files["ground_truths"], "w", encoding="utf-8") as f:
        json.dump(ground_truths_data, f, ensure_ascii=False, indent=2)
    logging.info(f"真实答案文件保存到: {files['ground_truths']}")

    return {k: str(v) for k, v in files.items()}


def create_config_file(file_paths: Dict[str, str], output_dir: Path) -> str:
    """
    创建配置文件

    Args:
        file_paths: 数据文件路径
        output_dir: 输出目录

    Returns:
        str: 配置文件路径
    """
    config_path = output_dir / "rag-dataset-1200_config.yaml"

    # 获取相对路径（相对于evaluation目录）
    eval_dir = Path(__file__).parent.parent

    try:
        questions_rel = Path(file_paths["questions"]).relative_to(eval_dir)
        contexts_rel = Path(file_paths["contexts"]).relative_to(eval_dir)
        ground_truths_rel = Path(file_paths["ground_truths"]).relative_to(eval_dir)
    except ValueError:
        # 如果无法获取相对路径，使用绝对路径
        questions_rel = file_paths["questions"]
        contexts_rel = file_paths["contexts"]
        ground_truths_rel = file_paths["ground_truths"]

    config_content = f"""# RAG-Dataset-1200评估配置文件
# 数据来源: https://huggingface.co/datasets/neural-bridge/rag-dataset-1200
evaluation:
  # 项目元信息
  project_name: "RAG-Dataset-1200评估"
  description: "基于Hugging Face neural-bridge/rag-dataset-1200数据集的RAG系统性能评估"
  version: "1.0.0"

  # 数据源配置
  dataset:
    # 数据源类型: "file"
    type: "file"

    # 文件模式配置
    questions_path: "{questions_rel}"
    ground_truths_path: "{ground_truths_rel}"
    contexts_path: "{contexts_rel}"

  # LLM配置（用于答案生成）
  llm_config:
    supplier: "siliconflow" # openai, siliconflow, volces, ollama, oneapi
    model: "deepseek-ai/DeepSeek-V3"
    api_key: "${{SILICONFLOW_API_KEY}}" # 从环境变量读取
    temperature: 0.1

  # 评估指标配置
  metrics:
    - "ContextRelevance" # 上下文相关性
    - "ContextPrecision" # 上下文精确度
    - "ContextRecall" # 上下文召回率
    - "Faithfulness" # 忠实度
    - "AnswerRelevancy" # 答案相关性

  # 评估器配置
  evaluator_config:
    # 用于评估的LLM配置（可以与答案生成不同）
    judge_llm:
      supplier: "siliconflow"
      model: "deepseek-ai/DeepSeek-V3"
      api_key: "${{SILICONFLOW_API_KEY}}"
      temperature: 0.0

  # 输出配置
  output:
    results_dir: "evaluation/results/rag-dataset-1200"
    export_format: ["json", "csv", "html"] # 支持的导出格式
    save_individual_scores: true
    generate_charts: true

# 数据集统计信息
dataset_info:
  total_samples: 1200
  train_samples: 960
  test_samples: 240
  source: "neural-bridge/rag-dataset-1200"
  description: |
    该数据集包含1200个RAG评估样本，每个样本包含:
    - context: 从Falcon RefinedWeb获取的上下文文档
    - question: 基于上下文生成的问题
    - answer: GPT-4生成的标准答案
    
    数据集特点:
    - 高质量的上下文-问题-答案三元组
    - 涵盖多种主题和领域
    - 适合RAG系统的全面评估
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    logging.info(f"配置文件保存到: {config_path}")
    return str(config_path)


def generate_statistics(df: pd.DataFrame, output_dir: Path):
    """
    生成数据集统计信息

    Args:
        df: 数据集DataFrame
        output_dir: 输出目录
    """
    stats = {
        "总样本数": len(df),
        "问题平均长度": df["question"].str.len().mean(),
        "答案平均长度": df["answer"].str.len().mean(),
        "上下文平均长度": df["context"].str.len().mean(),
        "问题最短长度": df["question"].str.len().min(),
        "问题最长长度": df["question"].str.len().max(),
        "答案最短长度": df["answer"].str.len().min(),
        "答案最长长度": df["answer"].str.len().max(),
        "上下文最短长度": df["context"].str.len().min(),
        "上下文最长长度": df["context"].str.len().max(),
    }

    # 保存统计信息
    stats_file = output_dir / "rag-dataset-1200_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logging.info("数据集统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            logging.info(f"  {key}: {value:.2f}")
        else:
            logging.info(f"  {key}: {value}")

    logging.info(f"详细统计信息保存到: {stats_file}")


def main():
    """主函数"""
    setup_logging()

    try:
        # 设置输出目录
        output_dir = Path(__file__).parent.parent / "datasets" / "rag-dataset-1200"

        logging.info("开始导入RAG-Dataset-1200...")

        # 下载数据集
        df = download_dataset()

        # 验证数据集
        if not validate_dataset(df):
            logging.error("数据集验证失败")
            return 1

        # 拆分数据集
        file_paths = split_dataset(df, output_dir)

        # 创建配置文件
        config_path = create_config_file(file_paths, output_dir)

        # 生成统计信息
        generate_statistics(df, output_dir)

        logging.info("数据集导入完成！")
        logging.info("生成的文件:")
        for file_type, path in file_paths.items():
            logging.info(f"  {file_type}: {path}")
        logging.info(f"  配置文件: {config_path}")

        return 0

    except Exception as e:
        logging.error(f"导入过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
