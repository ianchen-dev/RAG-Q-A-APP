"""
Integration tests for logging_settings.py module.

These tests verify logging configuration and require:
1. Write permissions for log directory creation
2. No external dependencies

Run with: uv run pytest test/integration/config/test_logging_settings.py -v
"""

import os

import pytest

from src.config.logging_settings import (
    LoggingSettings,
    get_logging_settings,
    LogLevel,
)


@pytest.mark.integration
class TestLoggingSettingsDefaults:
    """Integration tests for LoggingSettings default values."""

    def test_default_settings(self) -> None:
        """Test creating LoggingSettings with defaults."""
        settings = LoggingSettings()

        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.log_dir == "./log"
        assert settings.app_name == "fastapi_app"
        assert isinstance(settings.console_output, bool)
        assert isinstance(settings.file_logging, bool)
        assert isinstance(settings.error_file, bool)
        assert isinstance(settings.date_rotation, bool)
        assert settings.max_file_size > 0
        assert settings.backup_count >= 0

    def test_get_file_format(self) -> None:
        """Test getting file log format."""
        settings = LoggingSettings()
        file_format = settings.get_file_format()

        assert isinstance(file_format, str)
        assert "%(asctime)s" in file_format
        assert "%(levelname)s" in file_format
        assert "%(message)s" in file_format

    def test_get_console_format(self) -> None:
        """Test getting console log format."""
        settings = LoggingSettings()
        console_format = settings.get_console_format()

        assert isinstance(console_format, str)
        assert "%(asctime)s" in console_format
        assert "%(levelname)s" in console_format
        assert "%(message)s" in console_format


@pytest.mark.integration
class TestLoggingSettingsValidation:
    """Integration tests for LoggingSettings validation."""

    def test_validate_valid_settings(self) -> None:
        """Test validating valid settings."""
        settings = LoggingSettings()
        assert settings.validate() is True

    def test_validate_invalid_log_level(self) -> None:
        """Test validation fails with invalid log level."""
        # Temporarily set invalid log level
        settings = LoggingSettings()
        settings.log_level = "INVALID"  # type: ignore

        with pytest.raises(ValueError, match="无效的日志级别"):
            settings.validate()

    def test_validate_invalid_max_file_size(self) -> None:
        """Test validation fails with invalid max file size."""
        settings = LoggingSettings()
        settings.max_file_size = -1

        with pytest.raises(ValueError, match="文件大小必须大于0"):
            settings.validate()

    def test_validate_invalid_backup_count(self) -> None:
        """Test validation fails with invalid backup count."""
        settings = LoggingSettings()
        settings.backup_count = -1

        with pytest.raises(ValueError, match="备份数量不能小于0"):
            settings.validate()


@pytest.mark.integration
class TestLoggingSettingsStringRepresentation:
    """Integration tests for LoggingSettings string representation."""

    def test_str_representation(self) -> None:
        """Test string representation of settings."""
        settings = LoggingSettings()
        str_repr = str(settings)

        assert "LoggingSettings(" in str_repr
        assert "log_level=" in str_repr
        assert "log_dir=" in str_repr
        assert "app_name=" in str_repr


@pytest.mark.integration
class TestLoggingSettingsSingleton:
    """Integration tests for global logging settings instance."""

    def test_get_logging_settings(self) -> None:
        """Test getting global logging settings instance."""
        settings = get_logging_settings()

        assert isinstance(settings, LoggingSettings)
        assert settings is get_logging_settings()  # Same instance

    def test_logging_settings_global_instance(self) -> None:
        """Test global logging_settings instance."""
        from src.config.logging_settings import logging_settings

        assert isinstance(logging_settings, LoggingSettings)
