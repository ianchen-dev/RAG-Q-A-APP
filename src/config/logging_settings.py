"""
日志配置设置
通过环境变量控制日志行为的配置模块
"""

import os
from typing import Literal

# 日志级别类型定义
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingSettings:
    """日志配置设置类"""

    def __init__(self):
        """从环境变量初始化日志配置"""

        # 日志级别
        self.log_level: LogLevel = os.getenv("LOG_LEVEL", "INFO").upper()

        # 日志目录
        self.log_dir: str = os.getenv("LOG_DIR", "./log")

        # 应用名称
        self.app_name: str = os.getenv("APP_NAME", "fastapi_app")

        # 控制台输出
        self.console_output: bool = os.getenv("LOG_CONSOLE", "true").lower() == "true"

        # 文件日志配置
        self.file_logging: bool = os.getenv("LOG_FILE", "true").lower() == "true"

        # 错误日志单独文件
        self.error_file: bool = os.getenv("LOG_ERROR_FILE", "true").lower() == "true"

        # 按日期分割日志
        self.date_rotation: bool = (
            os.getenv("LOG_DATE_ROTATION", "true").lower() == "true"
        )

        # 文件大小轮转配置
        self.max_file_size: int = int(
            os.getenv("LOG_MAX_FILE_SIZE", "10485760")
        )  # 10MB
        self.backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

        # 时间轮转配置
        self.rotate_when: str = os.getenv("LOG_ROTATE_WHEN", "midnight")
        self.rotate_interval: int = int(os.getenv("LOG_ROTATE_INTERVAL", "1"))
        self.date_backup_count: int = int(os.getenv("LOG_DATE_BACKUP_COUNT", "30"))

        # 日志格式配置
        self.detailed_format: bool = (
            os.getenv("LOG_DETAILED_FORMAT", "true").lower() == "true"
        )

    def get_file_format(self) -> str:
        """获取文件日志格式"""
        if self.detailed_format:
            return "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        else:
            return "%(asctime)s - %(levelname)s - %(message)s"

    def get_console_format(self) -> str:
        """获取控制台日志格式"""
        return "%(asctime)s - %(levelname)s - %(message)s"

    def validate(self) -> bool:
        """验证配置有效性"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(
                f"无效的日志级别: {self.log_level}, 有效值: {valid_levels}"
            )

        if self.max_file_size <= 0:
            raise ValueError("文件大小必须大于0")

        if self.backup_count < 0:
            raise ValueError("备份数量不能小于0")

        return True

    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"""LoggingSettings(
    log_level={self.log_level},
    log_dir={self.log_dir},
    app_name={self.app_name},
    console_output={self.console_output},
    file_logging={self.file_logging},
    error_file={self.error_file},
    date_rotation={self.date_rotation},
    max_file_size={self.max_file_size},
    backup_count={self.backup_count}
)"""


# 全局配置实例
logging_settings = LoggingSettings()


def get_logging_settings() -> LoggingSettings:
    """获取日志配置实例"""
    return logging_settings


if __name__ == "__main__":
    # 测试配置
    settings = LoggingSettings()
    print("当前日志配置:")
    print(settings)

    # 验证配置
    try:
        settings.validate()
        print("\n✅ 配置验证通过")
    except ValueError as e:
        print(f"\n❌ 配置验证失败: {e}")
