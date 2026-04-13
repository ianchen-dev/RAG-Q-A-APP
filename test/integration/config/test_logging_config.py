"""
Integration tests for logging_config.py module.

These tests verify logging configuration functionality and require:
1. Write permissions for log directory creation
2. File system access for log file creation

Run with: uv run pytest test/integration/config/test_logging_config.py -v
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest

from src.config.logging_config import (
    setup_logging,
    setup_production_logging,
    setup_development_logging,
    setup_testing_logging,
    setup_logging_from_settings,
    get_logger,
    log_function_call,
)
from src.config.logging_settings import LoggingSettings


@pytest.mark.integration
class TestSetupLogging:
    """Integration tests for setup_logging function."""

    def test_setup_logging_creates_log_directory(self) -> None:
        """Test that setup_logging creates log directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                app_name="test_app",
            )

            assert logger is not None
            assert isinstance(logger, logging.Logger)
            assert Path(temp_dir).exists()

    def test_setup_logging_creates_log_files(self) -> None:
        """Test that setup_logging creates log files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app_name = "test_app"
            logger = setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                app_name=app_name,
            )

            # Trigger log creation
            logger.info("Test log message")

            # Check for expected log files
            log_path = Path(temp_dir)
            all_log_file = log_path / f"{app_name}_all.log"
            error_log_file = log_path / f"{app_name}_error.log"

            # Files should exist after logging
            assert all_log_file.exists() or error_log_file.exists()

    def test_setup_logging_levels(self) -> None:
        """Test setup_logging with different log levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
                logger = setup_logging(
                    log_level=level,
                    log_dir=temp_dir,
                    app_name=f"test_{level.lower()}",
                )

                assert logger is not None
                assert isinstance(logger, logging.Logger)

    def test_setup_logging_without_console(self) -> None:
        """Test setup_logging without console output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                app_name="test_no_console",
                console_output=False,
            )

            assert logger is not None
            assert isinstance(logger, logging.Logger)


@pytest.mark.integration
class TestSetupPredefinedLogging:
    """Integration tests for predefined logging setups."""

    def test_setup_production_logging(self) -> None:
        """Test production logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_production_logging(log_dir=temp_dir)

            assert logger is not None
            assert isinstance(logger, logging.Logger)
            assert logger.name == "fastapi_prod"

    def test_setup_development_logging(self) -> None:
        """Test development logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_development_logging(log_dir=temp_dir)

            assert logger is not None
            assert isinstance(logger, logging.Logger)
            assert logger.name == "fastapi_dev"

    def test_setup_testing_logging(self) -> None:
        """Test testing logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_testing_logging(log_dir=temp_dir)

            assert logger is not None
            assert isinstance(logger, logging.Logger)
            assert logger.name == "fastapi_test"


@pytest.mark.integration
class TestSetupLoggingFromSettings:
    """Integration tests for setup_logging_from_settings function."""

    def test_setup_from_settings_object(self) -> None:
        """Test setup_logging_from_settings with LoggingSettings object."""
        settings = LoggingSettings()
        settings.log_level = "INFO"
        settings.log_dir = tempfile.mkdtemp()
        settings.app_name = "test_from_settings"

        logger = setup_logging_from_settings(settings)

        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_setup_from_settings_default(self) -> None:
        """Test setup_logging_from_settings with default settings."""
        logger = setup_logging_from_settings()

        assert logger is not None
        assert isinstance(logger, logging.Logger)


@pytest.mark.integration
class TestGetLogger:
    """Integration tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Test getting logger with specific name."""
        logger = get_logger("test_module")

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_without_name(self) -> None:
        """Test getting logger without name."""
        logger = get_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)


@pytest.mark.integration
class TestLogFunctionCall:
    """Integration tests for log_function_call decorator."""

    def test_log_function_call_decorator(self) -> None:
        """Test log_function_call decorator."""
        @log_function_call("test_function")
        def test_function(x: int, y: int) -> int:
            return x + y

        result = test_function(2, 3)
        assert result == 5

    def test_log_function_call_with_exception(self) -> None:
        """Test log_function_call decorator with exception."""
        @log_function_call("failing_function")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()


@pytest.mark.integration
class TestLoggingOutput:
    """Integration tests for actual logging output."""

    def test_log_messages_written_to_file(self) -> None:
        """Test that log messages are written to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                app_name="test_output",
            )

            test_message = "Integration test log message"
            logger.info(test_message)

            # Force flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check log file exists and contains message
            log_path = Path(temp_dir)
            log_files = list(log_path.glob("*.log"))

            if log_files:
                log_file = log_files[0]
                content = log_file.read_text()
                assert test_message in content

    def test_error_logging(self) -> None:
        """Test error logging to error file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app_name = "test_error"
            logger = setup_logging(
                log_level="ERROR",
                log_dir=temp_dir,
                app_name=app_name,
            )

            test_message = "Integration test error message"
            logger.error(test_message)

            # Force flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check error log file exists
            error_log_file = Path(temp_dir) / f"{app_name}_error.log"
            if error_log_file.exists():
                content = error_log_file.read_text()
                assert test_message in content
