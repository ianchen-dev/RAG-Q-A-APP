"""
日志配置模块
提供统一的日志配置功能，支持文件和控制台输出
"""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logging_settings import LoggingSettings, get_logging_settings


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./log",
    app_name: str = "fastapi_app",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    配置应用程序的日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志目录路径
        app_name: 应用名称，用于日志文件命名
        max_file_size: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        console_output: 是否输出到控制台

    Returns:
        配置好的 logger 实例
    """

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 获取根记录器
    root_logger = logging.getLogger()

    # 如果已经配置过，先清除现有的处理器
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 设置日志级别
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    root_logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))

    # 创建日志格式器
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 简化的控制台格式器
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # 1. 配置控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # 2. 配置总日志文件处理器（所有级别）
    all_log_file = log_path / f"{app_name}_all.log"
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(all_log_file),
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 3. 配置错误日志文件处理器（仅错误和严重错误）
    error_log_file = log_path / f"{app_name}_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=str(error_log_file),
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # 4. 配置按日期分割的日志处理器
    date_log_file = log_path / f"{app_name}_{datetime.now().strftime('%Y-%m-%d')}.log"
    date_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(date_log_file),
        when="midnight",
        interval=1,
        backupCount=30,  # 保留30天的日志
        encoding="utf-8",
    )
    date_handler.setLevel(logging.INFO)
    date_handler.setFormatter(formatter)
    # 设置日志文件后缀
    date_handler.suffix = "%Y-%m-%d"
    root_logger.addHandler(date_handler)

    # 返回应用程序主logger
    app_logger = logging.getLogger(app_name)

    # 记录日志配置完成信息
    app_logger.info("日志系统初始化完成")
    app_logger.info(f"日志目录: {log_path.absolute()}")
    app_logger.info(f"日志级别: {log_level}")
    app_logger.info(f"控制台输出: {'启用' if console_output else '禁用'}")

    return app_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称，默认为调用模块名

    Returns:
        logger实例
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """
    装饰器：记录函数调用日志

    Usage:
        @log_function_call
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"调用函数: {func_name}, 参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数 {func_name} 执行完成")
                return result
            except Exception as e:
                logger.error(f"函数 {func_name} 执行出错: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


# 预定义的日志配置
def setup_production_logging(log_dir: str = "./log") -> logging.Logger:
    """生产环境日志配置"""
    return setup_logging(
        log_level="INFO",
        log_dir=log_dir,
        app_name="fastapi_prod",
        console_output=False,  # 生产环境不输出到控制台
    )


def setup_development_logging(log_dir: str = "./log") -> logging.Logger:
    """开发环境日志配置"""
    return setup_logging(
        log_level="DEBUG",
        log_dir=log_dir,
        app_name="fastapi_dev",
        console_output=True,  # 开发环境输出到控制台
    )


def setup_testing_logging(log_dir: str = "./log") -> logging.Logger:
    """测试环境日志配置"""
    return setup_logging(
        log_level="WARNING",
        log_dir=log_dir,
        app_name="fastapi_test",
        console_output=True,
    )


def setup_logging_from_settings(
    settings: Optional[LoggingSettings] = None,
) -> logging.Logger:
    """
    根据配置文件或环境变量设置日志

    Args:
        settings: 日志配置对象，如果为None则从环境变量获取

    Returns:
        配置好的 logger 实例
    """
    if settings is None:
        settings = get_logging_settings()

    # 验证配置
    settings.validate()

    return setup_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        app_name=settings.app_name,
        max_file_size=settings.max_file_size,
        backup_count=settings.backup_count,
        console_output=settings.console_output,
    )


if __name__ == "__main__":
    # 测试日志配置
    logger = setup_development_logging()

    logger.debug("这是一条DEBUG日志")
    logger.info("这是一条INFO日志")
    logger.warning("这是一条WARNING日志")
    logger.error("这是一条ERROR日志")
    logger.critical("这是一条CRITICAL日志")

    print("日志配置测试完成，请检查 ./log 目录中的日志文件")
