"""应用配置模块"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """配置类 - 平铺结构，所有配置项通过环境变量读取"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========================================
    # 日志配置
    # ========================================
    log_level: str = "INFO"

    # ========================================
    # 接口文档配置
    # ========================================
    api_docs_enabled: bool = True  # 开启 API 文档

    # ========================================
    # 服务配置
    # ========================================
    host: str = "0.0.0.0"
    port: int = 8082
    workers: int = 1

    # ========================================
    # 数据库配置
    # ========================================
    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/designer_db"
    )
    database_echo: bool = False  # 是否输出 SQL 日志

    # ========================================
    # CORS配置
    # ========================================
    cors_allowed_origin_patterns: list[str] = ["*"]
    cors_allowed_methods: list[str] = ["*"]
    cors_allowed_headers: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_max_age: int = 86400
    cors_expose_headers: list[str] = []
    
settings = Settings()

