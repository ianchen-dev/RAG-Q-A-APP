"""
数据加载器模块
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.config_schema import DatasetConfig


@dataclass
class EvaluationDataset:
    """评估数据集"""

    questions: List[str]
    ground_truths: List[List[str]]
    contexts: Optional[List[List[str]]] = None


class DataLoader:
    """统一的数据加载器"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def load_dataset(self) -> EvaluationDataset:
        """根据配置加载数据集

        Returns:
            EvaluationDataset: 加载的数据集

        Raises:
            ValueError: 不支持的数据源类型
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        self.logger.info(f"开始加载数据集，类型: {self.config.type}")

        if self.config.type == "file":
            return await self._load_from_files()
        elif self.config.type == "knowledge_base":
            return await self._load_from_knowledge_base()
        else:
            raise ValueError(f"不支持的数据源类型: {self.config.type}")

    async def _load_from_files(self) -> EvaluationDataset:
        """从文件加载数据

        Returns:
            EvaluationDataset: 从文件加载的数据集
        """
        self.logger.info("从文件加载数据...")

        # 加载问题
        questions_file = Path(self.config.questions_path)
        if not questions_file.exists():
            raise FileNotFoundError(f"问题文件不存在: {self.config.questions_path}")

        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"问题文件JSON格式错误: {e}")

        # 支持两种格式: {"questions": [...]} 或 [...]
        if isinstance(questions_data, dict):
            questions = questions_data.get("questions", [])
        elif isinstance(questions_data, list):
            questions = questions_data
        else:
            raise ValueError("问题文件格式错误，应为数组或包含questions字段的对象")

        # 加载真实答案
        gt_file = Path(self.config.ground_truths_path)
        if not gt_file.exists():
            raise FileNotFoundError(
                f"真实答案文件不存在: {self.config.ground_truths_path}"
            )

        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"真实答案文件JSON格式错误: {e}")

        # 支持两种格式: {"ground_truths": [...]} 或 [...]
        if isinstance(gt_data, dict):
            ground_truths = gt_data.get("ground_truths", [])
        elif isinstance(gt_data, list):
            ground_truths = gt_data
        else:
            raise ValueError(
                "真实答案文件格式错误，应为数组或包含ground_truths字段的对象"
            )

        # 验证数据一致性
        if len(questions) != len(ground_truths):
            raise ValueError(
                f"问题数量({len(questions)})与真实答案数量({len(ground_truths)})不匹配"
            )

        # 确保ground_truths中每个元素都是列表
        processed_ground_truths = []
        for i, gt in enumerate(ground_truths):
            if isinstance(gt, str):
                processed_ground_truths.append([gt])
            elif isinstance(gt, list):
                processed_ground_truths.append(gt)
            else:
                raise ValueError(f"真实答案第{i + 1}项格式错误，应为字符串或字符串数组")

        # 可选：加载上下文
        contexts = None
        if self.config.contexts_path:
            contexts_file = Path(self.config.contexts_path)
            if contexts_file.exists():
                try:
                    with open(contexts_file, "r", encoding="utf-8") as f:
                        contexts_data = json.load(f)

                    # 支持两种格式: {"contexts": [...]} 或 [...]
                    if isinstance(contexts_data, dict):
                        contexts = contexts_data.get("contexts", [])
                    elif isinstance(contexts_data, list):
                        contexts = contexts_data

                    # 验证上下文数据
                    if contexts and len(contexts) != len(questions):
                        self.logger.warning(
                            f"上下文数量({len(contexts)})与问题数量({len(questions)})不匹配，将忽略上下文"
                        )
                        contexts = None

                    # 确保contexts中每个元素都是列表
                    if contexts:
                        processed_contexts = []
                        for i, ctx in enumerate(contexts):
                            if isinstance(ctx, str):
                                processed_contexts.append([ctx])
                            elif isinstance(ctx, list):
                                processed_contexts.append(ctx)
                            else:
                                self.logger.warning(
                                    f"上下文第{i + 1}项格式错误，将忽略"
                                )
                                processed_contexts.append([])
                        contexts = processed_contexts

                except json.JSONDecodeError as e:
                    self.logger.warning(f"上下文文件JSON格式错误，将忽略: {e}")
                    contexts = None
            else:
                self.logger.warning(f"上下文文件不存在: {self.config.contexts_path}")

        self.logger.info(f"文件数据加载完成: {len(questions)}个问题")

        return EvaluationDataset(
            questions=questions,
            ground_truths=processed_ground_truths,
            contexts=contexts,
        )

    async def _load_from_knowledge_base(self) -> EvaluationDataset:
        """从知识库加载数据

        注意：这个方法需要与Knowledge.py集成，当前返回空数据

        Returns:
            EvaluationDataset: 从知识库加载的数据集
        """
        self.logger.info("从知识库加载数据...")

        # TODO: 实现从知识库加载逻辑
        # 这里需要集成Knowledge.py的逻辑来获取问题和真实答案
        # 目前返回空数据作为占位符

        raise NotImplementedError("知识库数据加载功能尚未实现，请使用文件模式")

        # 示例实现框架：
        # kb_config = self.config.knowledge_base
        # questions = await self._fetch_questions_from_kb(kb_config.kb_id)
        # ground_truths = await self._fetch_ground_truths_from_kb(kb_config.kb_id)
        #
        # return EvaluationDataset(
        #     questions=questions,
        #     ground_truths=ground_truths,
        #     contexts=None  # 将在答案生成时动态获取
        # )
