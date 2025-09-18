"""
向量数据库配置模块
支持Chroma和Milvus向量数据库的配置管理
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VectorDBType(str, Enum):
    """支持的向量数据库类型"""

    CHROMA = "chroma"
    MILVUS = "milvus"


class ChromaConfig(BaseModel):
    """Chroma数据库配置"""

    persist_directory: str = Field(default="chroma/", description="Chroma持久化目录")
    collection_metadata: Dict[str, Any] = Field(
        default={"hnsw:space": "cosine"}, description="Chroma集合元数据"
    )


class MilvusConfig(BaseModel):
    """Milvus数据库配置"""

    host: str = Field(default="localhost", description="Milvus服务器地址")
    port: int = Field(default=19530, description="Milvus服务器端口")
    user: Optional[str] = Field(default="", description="Milvus用户名")
    password: Optional[str] = Field(default="", description="Milvus密码")
    db_name: str = Field(default="default", description="Milvus数据库名称")
    secure: bool = Field(default=False, description="是否使用安全连接")
    index_type: str = Field(default="IVF_FLAT", description="索引类型")
    metric_type: str = Field(default="L2", description="距离度量类型")
    nlist: int = Field(default=1024, description="聚类中心数量")


class VectorDBConfig(BaseModel):
    """向量数据库总配置"""

    db_type: VectorDBType = Field(
        default=VectorDBType.CHROMA, description="向量数据库类型"
    )
    chroma: ChromaConfig = Field(default_factory=ChromaConfig, description="Chroma配置")
    milvus: MilvusConfig = Field(default_factory=MilvusConfig, description="Milvus配置")


# 全局配置实例
_vector_db_config: Optional[VectorDBConfig] = None


def load_vector_db_config() -> VectorDBConfig:
    """
    从环境变量和配置文件加载向量数据库配置
    """
    global _vector_db_config

    if _vector_db_config is not None:
        return _vector_db_config

    # 读取环境变量
    db_type = os.getenv("DEFAULT_VECTOR_DB_TYPE", "chroma").lower()

    # Chroma 配置
    chroma_config = ChromaConfig(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma/"),
        collection_metadata=_parse_metadata(
            os.getenv("CHROMA_COLLECTION_METADATA", '{"hnsw:space": "cosine"}')
        ),
    )

    # Milvus 配置
    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
        user=os.getenv("MILVUS_USER", ""),
        password=os.getenv("MILVUS_PASSWORD", ""),
        db_name=os.getenv("MILVUS_DB_NAME", "default"),
        secure=os.getenv("MILVUS_SECURE", "false").lower() == "true",
        index_type=os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT"),
        metric_type=os.getenv("MILVUS_METRIC_TYPE", "L2"),
        nlist=int(os.getenv("MILVUS_NLIST", "1024")),
    )

    _vector_db_config = VectorDBConfig(
        db_type=VectorDBType(db_type), chroma=chroma_config, milvus=milvus_config
    )

    logger.info(f"向量数据库配置加载完成: 类型={_vector_db_config.db_type}")
    return _vector_db_config


def get_vector_db_config() -> VectorDBConfig:
    """获取向量数据库配置"""
    if _vector_db_config is None:
        return load_vector_db_config()
    return _vector_db_config


def _parse_metadata(metadata_str: str) -> Dict[str, Any]:
    """解析元数据字符串"""
    try:
        import json

        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"无法解析元数据字符串: {metadata_str}, 使用默认值")
        return {"hnsw:space": "cosine"}
