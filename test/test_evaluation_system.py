"""
RAG评估系统测试用例
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.config.config_schema import EvaluationConfig, load_config
from evaluation.config.validator import ConfigValidator
from evaluation.core.answer_generator import AnswerGenerator
from evaluation.core.data_loader import DataLoader, EvaluationDataset
from evaluation.core.evaluator import RAGASEvaluator
from evaluation.core.main_evaluator import MainEvaluator


class TestConfigManagement:
    """配置管理测试"""

    def test_load_valid_config(self):
        """测试加载有效配置"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        assert isinstance(config, EvaluationConfig)
        assert config.project_name == "RAG系统评估示例"
        assert config.dataset.type == "file"
        assert "AnswerRelevancy" in config.metrics

    def test_config_validation(self):
        """测试配置验证"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        validator = ConfigValidator()
        is_valid = validator.validate(config)

        # 应该验证通过（可能有警告）
        assert is_valid or len(validator.errors) == 0

    def test_invalid_config_path(self):
        """测试无效配置文件路径"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestDataLoader:
    """数据加载器测试"""

    @pytest.mark.asyncio
    async def test_load_file_dataset(self):
        """测试从文件加载数据集"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        data_loader = DataLoader(config.dataset)
        dataset = await data_loader.load_dataset()

        assert isinstance(dataset, EvaluationDataset)
        assert len(dataset.questions) > 0
        assert len(dataset.ground_truths) > 0
        assert len(dataset.questions) == len(dataset.ground_truths)

    @pytest.mark.asyncio
    async def test_load_invalid_file(self):
        """测试加载无效文件"""
        from evaluation.config.config_schema import DatasetConfig

        invalid_config = DatasetConfig(
            type="file",
            questions_path="nonexistent_questions.json",
            ground_truths_path="nonexistent_truths.json",
        )

        data_loader = DataLoader(invalid_config)

        with pytest.raises(FileNotFoundError):
            await data_loader.load_dataset()


class TestAnswerGenerator:
    """答案生成器测试"""

    @pytest.mark.asyncio
    async def test_answer_generator_initialization(self):
        """测试答案生成器初始化"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        generator = AnswerGenerator(config)

        # 模拟环境下测试初始化
        with patch("evaluation.core.answer_generator.ChatSev") as mock_chat_sev:
            mock_chat_sev.return_value = MagicMock()
            await generator.initialize()

            assert generator.chat_service is not None

    @pytest.mark.asyncio
    async def test_answer_generation_mock(self):
        """测试答案生成（模拟模式）"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        generator = AnswerGenerator(config)

        # 模拟ChatSev的流式响应
        async def mock_stream_chat(*args, **kwargs):
            chunks = [
                {"type": "chunk", "data": "这是"},
                {"type": "chunk", "data": "一个"},
                {"type": "chunk", "data": "测试答案"},
            ]
            for chunk in chunks:
                yield chunk

        with patch("evaluation.core.answer_generator.ChatSev") as mock_chat_sev:
            mock_instance = MagicMock()
            mock_instance.stream_chat = mock_stream_chat
            mock_chat_sev.return_value = mock_instance

            await generator.initialize()

            questions = ["测试问题"]
            answers = await generator.generate_answers(questions)

            assert len(answers) == 1
            assert "测试答案" in answers[0]


class TestEvaluator:
    """评估器测试"""

    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        try:
            evaluator = RAGASEvaluator(config)
            assert evaluator is not None
            assert len(evaluator.selected_metrics) > 0
        except ImportError:
            # RAGAS未安装时跳过测试
            pytest.skip("RAGAS not available")

    @pytest.mark.asyncio
    async def test_fallback_evaluation(self):
        """测试备用评估模式"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        try:
            evaluator = RAGASEvaluator(config)

            # 强制使用备用评估
            questions = ["测试问题"]
            answers = ["测试答案"]
            contexts = [["测试上下文"]]
            ground_truths = [["真实答案"]]

            # 模拟RAGAS失败，触发备用评估
            with patch.object(
                evaluator, "evaluate", side_effect=Exception("RAGAS failed")
            ):
                result = evaluator._create_fallback_results(
                    questions, answers, contexts, ground_truths
                )

                assert "overall_scores" in result
                assert "detailed_results" in result
                assert result["metadata"]["is_fallback"] is True

        except ImportError:
            pytest.skip("RAGAS not available")


class TestMainEvaluator:
    """主评估器测试"""

    @pytest.mark.asyncio
    async def test_main_evaluator_initialization(self):
        """测试主评估器初始化"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        try:
            main_evaluator = MainEvaluator(config)
            assert main_evaluator is not None
            assert main_evaluator.config == config
        except ImportError:
            pytest.skip("RAGAS not available")

    @pytest.mark.asyncio
    async def test_quick_evaluation_mock(self):
        """测试快速评估（模拟模式）"""
        config_path = "evaluation/examples/sample_config.yaml"
        config = load_config(config_path)

        try:
            main_evaluator = MainEvaluator(config)

            # 模拟各个组件
            with (
                patch.object(main_evaluator.answer_generator, "initialize"),
                patch.object(
                    main_evaluator.answer_generator,
                    "generate_answers",
                    return_value=["测试答案1", "测试答案2"],
                ),
                patch.object(
                    main_evaluator.answer_generator,
                    "get_contexts_for_questions",
                    return_value=[["上下文1"], ["上下文2"]],
                ),
                patch.object(
                    main_evaluator.evaluator,
                    "evaluate",
                    return_value={
                        "overall_scores": {"AnswerRelevancy": 0.8},
                        "summary": {"average_score": 0.8},
                        "metadata": {"total_samples": 2},
                    },
                ),
            ):
                questions = ["问题1", "问题2"]
                expected_answers = [["答案1"], ["答案2"]]

                result = await main_evaluator.run_quick_evaluation(
                    questions, expected_answers
                )

                assert "overall_scores" in result
                assert "summary" in result

        except ImportError:
            pytest.skip("RAGAS not available")


class TestIntegration:
    """集成测试"""

    def test_example_files_exist(self):
        """测试示例文件是否存在"""
        assert Path("evaluation/examples/sample_config.yaml").exists()
        assert Path("evaluation/examples/sample_questions.json").exists()
        assert Path("evaluation/examples/sample_ground_truths.json").exists()

    def test_example_files_format(self):
        """测试示例文件格式"""
        # 测试问题文件
        with open(
            "evaluation/examples/sample_questions.json", "r", encoding="utf-8"
        ) as f:
            questions_data = json.load(f)
        assert "questions" in questions_data
        assert isinstance(questions_data["questions"], list)
        assert len(questions_data["questions"]) > 0

        # 测试真实答案文件
        with open(
            "evaluation/examples/sample_ground_truths.json", "r", encoding="utf-8"
        ) as f:
            gt_data = json.load(f)
        assert "ground_truths" in gt_data
        assert isinstance(gt_data["ground_truths"], list)
        assert len(gt_data["ground_truths"]) == len(questions_data["questions"])

    def test_directory_structure(self):
        """测试目录结构"""
        assert Path("evaluation/config").exists()
        assert Path("evaluation/core").exists()
        assert Path("evaluation/examples").exists()
        assert Path("evaluation/results").exists()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
