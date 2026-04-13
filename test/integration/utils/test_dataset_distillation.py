"""
Integration tests for dataset_distillation.py module.

These tests verify dataset distillation functionality and require:
1. Sample JSON files for questions and ground truths
2. Write permissions for output directory

Run with: uv run pytest test/integration/utils/test_dataset_distillation.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.dataset_distillation import (
    DatasetDistillation,
    DatasetSample,
)


@pytest.fixture
def sample_dataset() -> tuple[Path, Path]:
    """Create sample dataset files for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    questions_data = {
        "questions": [
            "What is machine learning?",
            "How does deep learning work?",
            "What is a neural network?",
            "Explain natural language processing.",
            "What is computer vision?",
            "How do decision trees work?",
            "What is reinforcement learning?",
            "Explain supervised learning.",
            "What is unsupervised learning?",
            "How does random forest work?",
        ]
    }

    ground_truths_data = {
        "ground_truths": [
            ["Machine learning is a subset of AI.", "ML enables computers to learn from data."],
            ["Deep learning uses neural networks.", "It automatically learns features."],
            ["Neural networks mimic brain structure.", "They consist of interconnected nodes."],
            ["NLP processes human language.", "It enables text understanding."],
            ["Computer vision enables image recognition.", "It analyzes visual data."],
            ["Decision trees split data recursively.", "They make predictions based on rules."],
            ["Reinforcement learning learns through rewards.", "It optimizes decision-making."],
            ["Supervised learning uses labeled data.", "It maps inputs to outputs."],
            ["Unsupervised learning finds patterns.", "It works with unlabeled data."],
            ["Random forest combines multiple trees.", "It improves prediction accuracy."],
        ]
    }

    questions_file = temp_dir / "questions.json"
    ground_truths_file = temp_dir / "ground_truths.json"

    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions_data, f)

    with open(ground_truths_file, "w", encoding="utf-8") as f:
        json.dump(ground_truths_data, f)

    return questions_file, ground_truths_file


@pytest.mark.integration
class TestDatasetDistillation:
    """Integration tests for DatasetDistillation class."""

    def test_load_dataset(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test loading dataset from files."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))

        distiller.load_dataset()

        assert len(distiller.samples) == 10
        assert distiller.total_samples == 10
        assert all(isinstance(s, DatasetSample) for s in distiller.samples)

    def test_mismatched_files_raises_error(self) -> None:
        """Test that mismatched question and answer counts raise error."""
        temp_dir = Path(tempfile.mkdtemp())

        questions_data = {"questions": ["Q1", "Q2"]}
        ground_truths_data = {"ground_truths": [["A1"]]}

        questions_file = temp_dir / "questions.json"
        ground_truths_file = temp_dir / "ground_truths.json"

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions_data, f)

        with open(ground_truths_file, "w", encoding="utf-8") as f:
            json.dump(ground_truths_data, f)

        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))

        with pytest.raises(ValueError, match="问题数量.*与答案数量.*不匹配"):
            distiller.load_dataset()


@pytest.mark.integration
class TestSamplingStrategies:
    """Integration tests for sampling strategies."""

    def test_random_sampling(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test random sampling strategy."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.random_sampling(sample_size=3, seed=42)

        assert len(sampled) == 3
        assert all(isinstance(s, DatasetSample) for s in sampled)

    def test_random_sampling_larger_than_dataset(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test random sampling when sample size exceeds dataset size."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.random_sampling(sample_size=100, seed=42)

        assert len(sampled) == 10  # Should return all samples

    def test_uniform_sampling(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test uniform sampling strategy."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.uniform_sampling(sample_size=3, seed=42)

        assert len(sampled) == 3
        assert all(isinstance(s, DatasetSample) for s in sampled)

    def test_diversity_sampling(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test diversity sampling strategy."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.diversity_sampling(sample_size=3, seed=42)

        assert len(sampled) == 3
        assert all(isinstance(s, DatasetSample) for s in sampled)


@pytest.mark.integration
class TestSaveSubset:
    """Integration tests for saving subset functionality."""

    def test_save_subset(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test saving subset to files."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.random_sampling(sample_size=3, seed=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            q_file, gt_file = distiller.save_subset(
                samples=sampled,
                output_dir=temp_dir,
                subset_name="test_subset",
            )

            assert Path(q_file).exists()
            assert Path(gt_file).exists()

            # Verify file contents
            with open(q_file, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            assert len(q_data["questions"]) == 3

            with open(gt_file, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            assert len(gt_data["ground_truths"]) == 3


@pytest.mark.integration
class TestStatistics:
    """Integration tests for statistics generation."""

    def test_generate_statistics(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test generating statistics for samples."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        sampled = distiller.random_sampling(sample_size=3, seed=42)
        stats = distiller.generate_statistics(sampled)

        assert "sample_count" in stats
        assert stats["sample_count"] == 3
        assert "question_length" in stats
        assert "question_complexity" in stats

    def test_generate_statistics_empty(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test generating statistics for empty sample list."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))

        stats = distiller.generate_statistics([])

        assert stats == {}


@pytest.mark.integration
class TestDistillDatasets:
    """Integration tests for batch dataset distillation."""

    def test_distill_datasets(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test batch dataset distillation."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))

        with tempfile.TemporaryDirectory() as temp_dir:
            results = distiller.distill_datasets(
                sample_sizes=[3, 5],
                output_dir=temp_dir,
                sampling_strategy="random",
                seed=42,
            )

            assert "dataset_3" in results
            assert "dataset_5" in results

            # Check files were created
            assert Path(results["dataset_3"]["questions_file"]).exists()
            assert Path(results["dataset_5"]["questions_file"]).exists()

    def test_distill_datasets_invalid_strategy(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test distill_datasets with invalid sampling strategy."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="不支持的采样策略"):
                distiller.distill_datasets(
                    sample_sizes=[3],
                    output_dir=temp_dir,
                    sampling_strategy="invalid_strategy",
                )


@pytest.mark.integration
class TestComplexityCalculation:
    """Integration tests for question complexity calculation."""

    def test_calculate_complexity(self, sample_dataset: tuple[Path, Path]) -> None:
        """Test complexity calculation for questions."""
        questions_file, ground_truths_file = sample_dataset
        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        # Get first sample
        sample = distiller.samples[0]

        # Complexity should be between 0 and 1
        assert 0 <= sample.question_complexity <= 1

    def test_complexity_increases_with_length(self) -> None:
        """Test that longer questions tend to have higher complexity."""
        short_question = "What is AI?"
        long_question = "According to the research specifically mentioned in the paper, how does the analysis compare particularly when described in detail?"

        temp_dir = Path(tempfile.mkdtemp())

        questions_data = {
            "questions": [short_question, long_question]
        }
        ground_truths_data = {
            "ground_truths": [["Short answer"], ["Long answer with more details"]]
        }

        questions_file = temp_dir / "questions.json"
        ground_truths_file = temp_dir / "ground_truths.json"

        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(questions_data, f)

        with open(ground_truths_file, "w", encoding="utf-8") as f:
            json.dump(ground_truths_data, f)

        distiller = DatasetDistillation(str(questions_file), str(ground_truths_file))
        distiller.load_dataset()

        # Long question should have higher complexity
        short_complexity = distiller.samples[0].question_complexity
        long_complexity = distiller.samples[1].question_complexity

        # Note: This is not always true due to how complexity is calculated,
        # but generally longer questions should have higher complexity
        assert long_complexity >= short_complexity
