#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集蒸馏工具 - Dataset Distillation Tool

该工具用于从大型问答数据集中提取小规模的子数据集，支持多种采样策略：
- 随机采样：完全随机选择
- 均匀分布采样：尽量保持数据在原数据集中的分布
- 多样性采样：基于问题长度和复杂度的多样性选择

作者: AI Assistant
创建时间: 2025-09-27
"""

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """数据样本类"""

    question_id: int
    question: str
    ground_truth: List[str]
    question_length: int
    question_complexity: float  # 基于问号、复杂词汇等计算的复杂度分数


class DatasetDistillation:
    """数据集蒸馏器"""

    def __init__(self, questions_file: str, ground_truths_file: str):
        """
        初始化数据集蒸馏器

        Args:
            questions_file: 问题文件路径
            ground_truths_file: 答案文件路径
        """
        self.questions_file = Path(questions_file)
        self.ground_truths_file = Path(ground_truths_file)
        self.samples: List[DatasetSample] = []
        self.total_samples = 0

    def load_dataset(self) -> None:
        """加载原始数据集"""
        logger.info("正在加载数据集...")

        # 加载问题
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        # 加载答案
        with open(self.ground_truths_file, "r", encoding="utf-8") as f:
            ground_truths_data = json.load(f)

        questions = questions_data.get("questions", [])
        ground_truths = ground_truths_data.get("ground_truths", [])

        if len(questions) != len(ground_truths):
            raise ValueError(
                f"问题数量({len(questions)})与答案数量({len(ground_truths)})不匹配"
            )

        self.total_samples = len(questions)
        logger.info(f"数据集总数: {self.total_samples}")

        # 创建样本对象
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            sample = DatasetSample(
                question_id=i,
                question=question,
                ground_truth=ground_truth,
                question_length=len(question.split()),
                question_complexity=self._calculate_complexity(question),
            )
            self.samples.append(sample)

        logger.info("数据集加载完成")

    def _calculate_complexity(self, question: str) -> float:
        """
        计算问题复杂度分数

        Args:
            question: 问题文本

        Returns:
            复杂度分数 (0-1之间)
        """
        # 基础分数
        score = 0.0

        # 长度因子 (归一化到0-0.3)
        word_count = len(question.split())
        length_score = min(word_count / 50.0, 1.0) * 0.3
        score += length_score

        # 问号数量 (多个问号表示复杂查询)
        question_marks = question.count("?")
        question_score = min(question_marks / 3.0, 1.0) * 0.2
        score += question_score

        # 复杂词汇检测
        complex_words = [
            "according",
            "specifically",
            "particularly",
            "mentioned",
            "described",
            "discussed",
            "analysis",
            "comparison",
        ]
        complex_word_count = sum(
            1 for word in complex_words if word.lower() in question.lower()
        )
        complex_score = min(complex_word_count / len(complex_words), 1.0) * 0.3
        score += complex_score

        # 连词和从句检测
        conjunctions = [
            "and",
            "or",
            "but",
            "however",
            "therefore",
            "because",
            "since",
            "although",
        ]
        conjunction_count = sum(
            1 for conj in conjunctions if conj.lower() in question.lower()
        )
        conjunction_score = min(conjunction_count / 5.0, 1.0) * 0.2
        score += conjunction_score

        return min(score, 1.0)

    def random_sampling(
        self, sample_size: int, seed: Optional[int] = None
    ) -> List[DatasetSample]:
        """
        随机采样

        Args:
            sample_size: 采样数量
            seed: 随机种子，用于结果复现

        Returns:
            采样结果列表
        """
        if seed is not None:
            random.seed(seed)

        if sample_size >= len(self.samples):
            logger.warning(
                f"采样数量({sample_size})大于等于总样本数({len(self.samples)})，返回全部样本"
            )
            return self.samples.copy()

        return random.sample(self.samples, sample_size)

    def uniform_sampling(
        self, sample_size: int, seed: Optional[int] = None
    ) -> List[DatasetSample]:
        """
        均匀分布采样 - 在原数据集中均匀选择样本

        Args:
            sample_size: 采样数量
            seed: 随机种子

        Returns:
            采样结果列表
        """
        if seed is not None:
            random.seed(seed)

        if sample_size >= len(self.samples):
            return self.samples.copy()

        # 计算采样间隔
        interval = len(self.samples) / sample_size
        indices = [int(i * interval) for i in range(sample_size)]

        # 添加一些随机性，避免过于机械
        for i in range(len(indices)):
            offset = random.randint(-int(interval * 0.1), int(interval * 0.1))
            indices[i] = max(0, min(len(self.samples) - 1, indices[i] + offset))

        # 去重并排序
        indices = sorted(list(set(indices)))

        # 如果去重后数量不足，补充随机样本
        if len(indices) < sample_size:
            remaining_indices = [
                i for i in range(len(self.samples)) if i not in indices
            ]
            additional_indices = random.sample(
                remaining_indices, sample_size - len(indices)
            )
            indices.extend(additional_indices)

        return [self.samples[i] for i in indices[:sample_size]]

    def diversity_sampling(
        self, sample_size: int, seed: Optional[int] = None
    ) -> List[DatasetSample]:
        """
        多样性采样 - 基于问题长度和复杂度的多样化选择

        Args:
            sample_size: 采样数量
            seed: 随机种子

        Returns:
            采样结果列表
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if sample_size >= len(self.samples):
            return self.samples.copy()

        # 按复杂度和长度分层
        complexity_bins = 5  # 复杂度分为5层
        length_bins = 5  # 长度分为5层

        # 计算分层边界
        complexities = [s.question_complexity for s in self.samples]
        lengths = [s.question_length for s in self.samples]

        complexity_percentiles = np.percentile(complexities, [20, 40, 60, 80])
        length_percentiles = np.percentile(lengths, [20, 40, 60, 80])

        # 创建分层字典
        bins = defaultdict(list)

        for sample in self.samples:
            # 确定复杂度层级
            comp_bin = 0
            for i, threshold in enumerate(complexity_percentiles):
                if sample.question_complexity > threshold:
                    comp_bin = i + 1

            # 确定长度层级
            len_bin = 0
            for i, threshold in enumerate(length_percentiles):
                if sample.question_length > threshold:
                    len_bin = i + 1

            bins[(comp_bin, len_bin)].append(sample)

        # 从每个bin中采样
        samples_per_bin = max(1, sample_size // len(bins))
        selected_samples = []

        for bin_key, bin_samples in bins.items():
            if not bin_samples:
                continue

            # 从当前bin中随机选择样本
            bin_sample_size = min(samples_per_bin, len(bin_samples))
            selected_samples.extend(random.sample(bin_samples, bin_sample_size))

        # 如果样本不足，从剩余样本中随机补充
        if len(selected_samples) < sample_size:
            remaining_samples = [s for s in self.samples if s not in selected_samples]
            additional_count = sample_size - len(selected_samples)
            if remaining_samples:
                additional_samples = random.sample(
                    remaining_samples, min(additional_count, len(remaining_samples))
                )
                selected_samples.extend(additional_samples)

        # 如果样本过多，随机删除多余的
        if len(selected_samples) > sample_size:
            selected_samples = random.sample(selected_samples, sample_size)

        return selected_samples

    def save_subset(
        self, samples: List[DatasetSample], output_dir: str, subset_name: str
    ) -> Tuple[str, str]:
        """
        保存子数据集

        Args:
            samples: 采样样本列表
            output_dir: 输出目录
            subset_name: 子集名称

        Returns:
            (questions_file_path, ground_truths_file_path)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 准备数据
        questions = [sample.question for sample in samples]
        ground_truths = [sample.ground_truth for sample in samples]

        # 保存问题文件
        questions_file = output_path / f"{subset_name}_questions.json"
        questions_data = {"questions": questions}

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)

        # 保存答案文件
        ground_truths_file = output_path / f"{subset_name}_ground_truths.json"
        ground_truths_data = {"ground_truths": ground_truths}

        with open(ground_truths_file, "w", encoding="utf-8") as f:
            json.dump(ground_truths_data, f, ensure_ascii=False, indent=2)

        logger.info(f"子数据集已保存: {subset_name} ({len(samples)}条)")
        logger.info(f"  问题文件: {questions_file}")
        logger.info(f"  答案文件: {ground_truths_file}")

        return str(questions_file), str(ground_truths_file)

    def generate_statistics(self, samples: List[DatasetSample]) -> Dict[str, Any]:
        """
        生成数据集统计信息

        Args:
            samples: 样本列表

        Returns:
            统计信息字典
        """
        if not samples:
            return {}

        complexities = [s.question_complexity for s in samples]
        lengths = [s.question_length for s in samples]

        stats = {
            "sample_count": len(samples),
            "question_length": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "median": np.median(lengths),
            },
            "question_complexity": {
                "mean": np.mean(complexities),
                "std": np.std(complexities),
                "min": min(complexities),
                "max": max(complexities),
                "median": np.median(complexities),
            },
        }

        return stats

    def distill_datasets(
        self,
        sample_sizes: List[int],
        output_dir: str,
        sampling_strategy: str = "diversity",
        seed: Optional[int] = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量蒸馏数据集

        Args:
            sample_sizes: 采样大小列表，如 [10, 100, 500]
            output_dir: 输出目录
            sampling_strategy: 采样策略 ('random', 'uniform', 'diversity')
            seed: 随机种子

        Returns:
            每个子集的信息字典
        """
        if not self.samples:
            self.load_dataset()

        results = {}

        # 选择采样方法
        sampling_methods = {
            "random": self.random_sampling,
            "uniform": self.uniform_sampling,
            "diversity": self.diversity_sampling,
        }

        if sampling_strategy not in sampling_methods:
            raise ValueError(f"不支持的采样策略: {sampling_strategy}")

        sampling_method = sampling_methods[sampling_strategy]

        logger.info(f"开始使用 {sampling_strategy} 策略蒸馏数据集...")

        for sample_size in sample_sizes:
            logger.info(f"正在生成 {sample_size} 条样本的子集...")

            # 采样
            sampled_data = sampling_method(sample_size, seed)

            # 保存
            subset_name = f"rag-dataset-{sample_size}"
            questions_file, ground_truths_file = self.save_subset(
                sampled_data, output_dir, subset_name
            )

            # 统计信息
            stats = self.generate_statistics(sampled_data)

            results[f"dataset_{sample_size}"] = {
                "sample_size": sample_size,
                "questions_file": questions_file,
                "ground_truths_file": ground_truths_file,
                "statistics": stats,
                "sampling_strategy": sampling_strategy,
            }

        # 保存汇总信息
        summary_file = Path(output_dir) / "distillation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"数据集蒸馏完成，汇总信息保存至: {summary_file}")

        return results


def main():
    """主函数 - 命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG数据集蒸馏工具")
    parser.add_argument("--questions", required=True, help="问题文件路径")
    parser.add_argument("--ground-truths", required=True, help="答案文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[10, 100, 500], help="采样大小列表"
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "uniform", "diversity"],
        default="diversity",
        help="采样策略",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 创建蒸馏器
    distiller = DatasetDistillation(args.questions, args.ground_truths)

    # 执行蒸馏
    results = distiller.distill_datasets(
        sample_sizes=args.sizes,
        output_dir=args.output_dir,
        sampling_strategy=args.strategy,
        seed=args.seed,
    )

    # 打印结果
    print("\n=== 数据集蒸馏结果 ===")
    for dataset_name, info in results.items():
        print(f"\n{dataset_name}:")
        print(f"  样本数量: {info['sample_size']}")
        print(f"  采样策略: {info['sampling_strategy']}")
        print(f"  问题文件: {info['questions_file']}")
        print(f"  答案文件: {info['ground_truths_file']}")

        stats = info["statistics"]
        print(f"  问题长度: 平均 {stats['question_length']['mean']:.1f} 词")
        print(f"  问题复杂度: 平均 {stats['question_complexity']['mean']:.3f}")


if __name__ == "__main__":
    main()
