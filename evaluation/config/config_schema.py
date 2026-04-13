"""
RAG评估系统配置模式定义
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


@dataclass
class LLMConfig:
    """LLM配置"""

    supplier: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_length: int = 2048


@dataclass
class RemoteRerankConfig:
    """远程重排序配置"""

    api_key: str
    model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankerConfig:
    """重排序配置"""

    use_reranker: bool = False
    reranker_type: Literal["local", "remote"] = "remote"
    remote_rerank_config: Optional[RemoteRerankConfig] = None
    rerank_top_n: int = 3


@dataclass
class KnowledgeConfig:
    """知识检索配置"""

    use_bm25: bool = False
    bm25_k: int = 3
    reranker_config: Optional[RerankerConfig] = None


@dataclass
class KnowledgeBaseConfig:
    """知识库配置"""

    kb_id: str
    search_k: int = 5
    filter_by_file_md5: Optional[str] = None


@dataclass
class DatasetConfig:
    """数据集配置"""

    type: Literal["file", "knowledge_base"]
    questions_path: Optional[str] = None
    ground_truths_path: Optional[str] = None
    contexts_path: Optional[str] = None
    knowledge_base: Optional[KnowledgeBaseConfig] = None


@dataclass
class JudgeLLMConfig:
    """评估用LLM配置"""

    supplier: str
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.0


@dataclass
class JudgeEmbeddingConfig:
    supplier: str
    model: str
    api_key: Optional[str] = None


@dataclass
class EvaluatorConfig:
    """评估器配置"""

    judge_llm: JudgeLLMConfig
    judge_embedding: JudgeEmbeddingConfig


@dataclass
class OutputConfig:
    """输出配置"""

    results_dir: str = "evaluation/results"
    export_format: List[str] = field(default_factory=lambda: ["json", "csv", "html"])
    save_individual_scores: bool = True
    generate_charts: bool = True


@dataclass
class EvaluationConfig:
    """完整评估配置"""

    project_name: str
    dataset: DatasetConfig
    llm_config: LLMConfig
    evaluator_config: EvaluatorConfig
    description: str = ""
    version: str = "1.0.0"
    knowledge_config: Optional[KnowledgeConfig] = None
    metrics: List[str] = field(
        default_factory=lambda: [
            "ContextRelevance",
            "ContextPrecision",
            "ContextRecall",
            "Faithfulness",
            "AnswerRelevancy",
        ]
    )
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str) -> EvaluationConfig:
    """从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        EvaluationConfig: 评估配置对象

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
        ValueError: 配置格式错误
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML格式错误: {e}")

    if "evaluation" not in config_dict:
        raise ValueError("配置文件必须包含 'evaluation' 根节点")

    eval_config = config_dict["evaluation"]

    # 处理环境变量替换
    eval_config = _replace_env_vars(eval_config)

    # 构建配置对象
    try:
        return _build_config(eval_config)
    except Exception as e:
        raise ValueError(f"配置构建失败: {e}")


def _replace_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """递归替换配置中的环境变量

    支持格式: ${ENV_VAR_NAME} 或 ${ENV_VAR_NAME:default_value}
    """
    if isinstance(config, dict):
        return {k: _replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        if config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            if ":" in env_var:
                var_name, default_value = env_var.split(":", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(env_var, config)
    return config


def _build_config(config_dict: Dict[str, Any]) -> EvaluationConfig:
    """构建配置对象"""

    # 构建DatasetConfig
    dataset_dict = config_dict["dataset"]
    knowledge_base = None
    if dataset_dict.get("knowledge_base"):
        kb_dict = dataset_dict["knowledge_base"]
        knowledge_base = KnowledgeBaseConfig(**kb_dict)

    dataset_config = DatasetConfig(
        type=dataset_dict["type"],
        questions_path=dataset_dict.get("questions_path"),
        ground_truths_path=dataset_dict.get("ground_truths_path"),
        contexts_path=dataset_dict.get("contexts_path"),
        knowledge_base=knowledge_base,
    )

    # 构建LLMConfig
    llm_dict = config_dict["llm_config"]
    llm_config = LLMConfig(**llm_dict)

    # 构建KnowledgeConfig (可选)
    knowledge_config = None
    if "knowledge_config" in config_dict:
        kc_dict = config_dict["knowledge_config"]

        # 构建RerankerConfig
        reranker_config = None
        if "reranker_config" in kc_dict:
            rc_dict = kc_dict["reranker_config"]
            remote_rerank_config = None
            if rc_dict.get("remote_rerank_config"):
                remote_rerank_config = RemoteRerankConfig(
                    **rc_dict["remote_rerank_config"]
                )

            reranker_config = RerankerConfig(
                use_reranker=rc_dict.get("use_reranker", False),
                reranker_type=rc_dict.get("reranker_type", "remote"),
                remote_rerank_config=remote_rerank_config,
                rerank_top_n=rc_dict.get("rerank_top_n", 3),
            )

        knowledge_config = KnowledgeConfig(
            use_bm25=kc_dict.get("use_bm25", False),
            bm25_k=kc_dict.get("bm25_k", 3),
            reranker_config=reranker_config,
        )

    # 构建EvaluatorConfig
    eval_dict = config_dict["evaluator_config"]
    judge_llm = JudgeLLMConfig(**eval_dict["judge_llm"])
    judge_embedding = JudgeEmbeddingConfig(**eval_dict["judge_embedding"])
    evaluator_config = EvaluatorConfig(
        judge_llm=judge_llm, judge_embedding=judge_embedding
    )

    # 构建OutputConfig
    output_config = OutputConfig()
    if "output" in config_dict:
        output_dict = config_dict["output"]
        output_config = OutputConfig(**output_dict)

    return EvaluationConfig(
        project_name=config_dict["project_name"],
        description=config_dict.get("description", ""),
        version=config_dict.get("version", "1.0.0"),
        dataset=dataset_config,
        llm_config=llm_config,
        knowledge_config=knowledge_config,
        metrics=config_dict.get(
            "metrics",
            [
                "ContextRelevance",
                "ContextPrecision",
                "ContextRecall",
                "Faithfulness",
                "AnswerRelevancy",
            ],
        ),
        evaluator_config=evaluator_config,
        output=output_config,
    )
