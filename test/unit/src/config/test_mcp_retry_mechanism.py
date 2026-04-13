"""
MCP重试机制测试用例

测试MCP客户端的重试机制，包括：
1. 重试配置测试
2. 指数退避算法测试
3. 错误分类测试
4. 重试逻辑测试
5. 优雅降级测试
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 导入待测试的模块
from src.config.mcp_client_manager import (
    ApplicationMCPClient,
    MCPRetryConfig,
    get_cached_mcp_tools,
    get_mcp_client_status,
    is_retryable_error,
    load_mcp_retry_config,
    mcp_health_check,
    retry_with_backoff,
    set_mcp_retry_config,
    startup_mcp_client,
)


@pytest.mark.unit


class TestMCPRetryConfig:
    """测试MCPRetryConfig配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = MCPRetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.jitter_range == 0.1

    def test_custom_config(self):
        """测试自定义配置"""
        config = MCPRetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            jitter_range=0.3,
        )
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.jitter_range == 0.3

    def test_delay_calculation_without_jitter(self):
        """测试不带抖动的延迟计算"""
        config = MCPRetryConfig(
            base_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False
        )

        # 测试指数退避
        assert config.get_delay(0) == 1.0  # 1 * 2^0 = 1
        assert config.get_delay(1) == 2.0  # 1 * 2^1 = 2
        assert config.get_delay(2) == 4.0  # 1 * 2^2 = 4
        assert config.get_delay(3) == 8.0  # 1 * 2^3 = 8
        assert config.get_delay(4) == 10.0  # max(16, 10) = 10

    def test_delay_calculation_with_jitter(self):
        """测试带抖动的延迟计算"""
        config = MCPRetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=True,
            jitter_range=0.2,
        )

        # 测试抖动范围
        for attempt in range(3):
            delay = config.get_delay(attempt)
            expected_base = min(1.0 * (2.0**attempt), 10.0)
            min_delay = expected_base * 0.8  # -20%
            max_delay = expected_base * 1.2  # +20%
            assert min_delay <= delay <= max_delay

    def test_load_config_from_env(self):
        """测试从环境变量加载配置"""
        # 设置测试环境变量
        test_env = {
            "MCP_RETRY_MAX_RETRIES": "5",
            "MCP_RETRY_BASE_DELAY": "2.5",
            "MCP_RETRY_MAX_DELAY": "60.0",
            "MCP_RETRY_EXPONENTIAL_BASE": "1.5",
            "MCP_RETRY_ENABLE_JITTER": "false",
            "MCP_RETRY_JITTER_RANGE": "0.3",
        }

        with patch.dict(os.environ, test_env):
            config = load_mcp_retry_config()
            assert config.max_retries == 5
            assert config.base_delay == 2.5
            assert config.max_delay == 60.0
            assert config.exponential_base == 1.5
            assert config.jitter is False
            assert config.jitter_range == 0.3


class TestErrorClassification:
    """测试错误分类功能"""

    def test_retryable_error_types(self):
        """测试可重试的错误类型"""
        # 可重试的错误类型
        assert is_retryable_error(ConnectionError("网络连接错误"))
        assert is_retryable_error(TimeoutError("连接超时"))
        assert is_retryable_error(OSError("系统错误"))
        # 暂时跳过McpError测试，因为需要特定的错误对象结构
        # assert is_retryable_error(McpError("MCP协议错误"))

    def test_retryable_error_messages(self):
        """测试基于错误消息的可重试判断"""
        retryable_messages = [
            "Connection refused",
            "Connection reset",
            "Connection closed",
            "Network timeout",
            "Service unavailable",
            "Temporary failure",
        ]

        for message in retryable_messages:
            error = Exception(message)
            assert is_retryable_error(error), f"应该可重试的错误: {message}"

    def test_non_retryable_errors(self):
        """测试不可重试的错误"""
        non_retryable_errors = [
            ValueError("参数错误"),
            TypeError("类型错误"),
            KeyError("键不存在"),
            Exception("未知错误"),
        ]

        for error in non_retryable_errors:
            assert not is_retryable_error(error), f"不应该重试的错误: {error}"


class TestRetryBackoff:
    """测试重试退避算法"""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self):
        """测试成功执行，无需重试"""
        mock_func = AsyncMock(return_value="success")

        result = await retry_with_backoff(
            mock_func, MCPRetryConfig(max_retries=3), "测试操作"
        )

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """测试可重试错误的重试逻辑"""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            ConnectionError("连接失败"),
            ConnectionError("连接失败"),
            "success",  # 第三次成功
        ]

        config = MCPRetryConfig(max_retries=3, base_delay=0.01, jitter=False)

        result = await retry_with_backoff(mock_func, config, "测试操作")

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        mock_func = AsyncMock(side_effect=ConnectionError("持续连接失败"))

        config = MCPRetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        with pytest.raises(ConnectionError, match="持续连接失败"):
            await retry_with_backoff(mock_func, config, "测试操作")

        assert mock_func.call_count == 3  # 1次初始尝试 + 2次重试

    @pytest.mark.asyncio
    async def test_non_retryable_error_no_retry(self):
        """测试不可重试错误，不进行重试"""
        mock_func = AsyncMock(side_effect=ValueError("参数错误"))

        config = MCPRetryConfig(max_retries=3, base_delay=0.01)

        with pytest.raises(ValueError, match="参数错误"):
            await retry_with_backoff(mock_func, config, "测试操作")

        assert mock_func.call_count == 1  # 只尝试一次

    @pytest.mark.asyncio
    async def test_retry_delay_timing(self):
        """测试重试延迟时间"""
        import time

        mock_func = AsyncMock()
        mock_func.side_effect = [
            ConnectionError("失败1"),
            ConnectionError("失败2"),
            "success",
        ]

        config = MCPRetryConfig(
            max_retries=3, base_delay=0.1, exponential_base=2.0, jitter=False
        )

        start_time = time.time()
        result = await retry_with_backoff(mock_func, config, "测试操作")
        end_time = time.time()

        assert result == "success"
        # 验证大致的延迟时间：0.1 + 0.2 = 0.3秒
        assert end_time - start_time >= 0.25  # 考虑执行时间误差


class TestMCPClientManager:
    """测试MCP客户端管理器"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 重置单例状态
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []
        ApplicationMCPClient._retry_config = MCPRetryConfig(
            max_retries=2, base_delay=0.01, jitter=False
        )

    def teardown_method(self):
        """每个测试方法后的清理"""
        # 清理单例状态
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []

    @pytest.mark.asyncio
    async def test_successful_startup(self):
        """测试成功启动"""
        with patch(
            "src.config.mcp_client_manager.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
            mock_client.get_tools.return_value = mock_tools
            mock_client_class.return_value = mock_client

            client_manager = ApplicationMCPClient()
            await client_manager.startup()

            assert ApplicationMCPClient._instance is not None
            assert len(ApplicationMCPClient._tools) == 2
            mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_with_retry_success(self):
        """测试启动过程中重试成功"""
        with patch(
            "src.config.mcp_client_manager.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_tools = [MagicMock(name="tool1")]

            # 模拟第一次失败，第二次成功
            mock_client_class.side_effect = [ConnectionError("连接失败"), mock_client]
            mock_client.get_tools.return_value = mock_tools

            client_manager = ApplicationMCPClient()
            await client_manager.startup()

            assert ApplicationMCPClient._instance is not None
            assert len(ApplicationMCPClient._tools) == 1
            assert mock_client_class.call_count == 2

    @pytest.mark.asyncio
    async def test_startup_failure_after_retries(self):
        """测试重试后仍然失败"""
        with patch(
            "src.config.mcp_client_manager.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client_class.side_effect = ConnectionError("持续连接失败")

            client_manager = ApplicationMCPClient()
            await client_manager.startup()

            # 验证优雅降级
            assert ApplicationMCPClient._instance is None
            assert len(ApplicationMCPClient._tools) == 0
            # 验证重试次数：1次初始尝试 + 2次重试 = 3次
            assert mock_client_class.call_count == 3

    @pytest.mark.asyncio
    async def test_startup_without_retry(self):
        """测试不使用重试机制的启动"""
        with patch(
            "src.config.mcp_client_manager.MultiServerMCPClient"
        ) as mock_client_class:
            mock_client_class.side_effect = ConnectionError("连接失败")

            client_manager = ApplicationMCPClient()
            await client_manager.startup(enable_retry=False)

            # 验证只尝试一次
            assert mock_client_class.call_count == 1
            assert ApplicationMCPClient._instance is None

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """测试健康检查成功"""
        mock_client = AsyncMock()
        mock_client.get_tools.return_value = [MagicMock()]
        ApplicationMCPClient._instance = mock_client

        client_manager = ApplicationMCPClient()
        result = await client_manager.health_check()

        assert result is True
        mock_client.get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """测试健康检查失败"""
        mock_client = AsyncMock()
        mock_client.get_tools.side_effect = ConnectionError("连接失败")
        ApplicationMCPClient._instance = mock_client

        client_manager = ApplicationMCPClient()
        result = await client_manager.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_client_status(self):
        """测试获取客户端状态"""
        mock_tools = [MagicMock(), MagicMock()]
        ApplicationMCPClient._instance = MagicMock()
        ApplicationMCPClient._tools = mock_tools

        client_manager = ApplicationMCPClient()
        status = await client_manager.get_client_status()

        assert status["initialized"] is True
        assert status["tools_count"] == 2
        assert status["tools_available"] is True
        assert "retry_config" in status


class TestPublicInterface:
    """测试公共接口函数"""

    def setup_method(self):
        """每个测试方法前的设置"""
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []

    def teardown_method(self):
        """每个测试方法后的清理"""
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []

    @pytest.mark.asyncio
    async def test_startup_mcp_client(self):
        """测试公共启动接口"""
        with patch.object(ApplicationMCPClient, "startup") as mock_startup:
            await startup_mcp_client(enable_retry=True)
            mock_startup.assert_called_once_with(enable_retry=True)

    @pytest.mark.asyncio
    async def test_get_cached_mcp_tools(self):
        """测试获取缓存工具接口"""
        with patch.object(ApplicationMCPClient, "get_mcp_tools") as mock_get_tools:
            mock_get_tools.return_value = ["tool1", "tool2"]

            tools = await get_cached_mcp_tools(enable_retry=False)

            assert tools == ["tool1", "tool2"]
            mock_get_tools.assert_called_once_with(enable_retry=False)

    @pytest.mark.asyncio
    async def test_mcp_health_check(self):
        """测试健康检查接口"""
        with patch.object(ApplicationMCPClient, "health_check") as mock_health_check:
            mock_health_check.return_value = True

            result = await mcp_health_check()

            assert result is True
            mock_health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mcp_client_status(self):
        """测试获取状态接口"""
        with patch.object(ApplicationMCPClient, "get_client_status") as mock_get_status:
            mock_status = {"initialized": True, "tools_count": 1}
            mock_get_status.return_value = mock_status

            status = await get_mcp_client_status()

            assert status == mock_status
            mock_get_status.assert_called_once()

    def test_set_mcp_retry_config(self):
        """测试设置重试配置"""
        with patch.object(ApplicationMCPClient, "set_retry_config") as mock_set_config:
            set_mcp_retry_config(max_retries=5, base_delay=2.0, max_delay=60.0)

            # 验证调用了设置方法
            mock_set_config.assert_called_once()
            # 验证传递的配置参数
            config_arg = mock_set_config.call_args[0][0]
            assert config_arg.max_retries == 5
            assert config_arg.base_delay == 2.0
            assert config_arg.max_delay == 60.0


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        """每个测试方法前的设置"""
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []

    def teardown_method(self):
        """每个测试方法后的清理"""
        ApplicationMCPClient._instance = None
        ApplicationMCPClient._tools = []

    @pytest.mark.asyncio
    async def test_full_startup_workflow(self):
        """测试完整的启动工作流"""
        with patch(
            "src.config.mcp_client_manager.MultiServerMCPClient"
        ) as mock_client_class:
            # 模拟一次失败，然后成功
            mock_client = AsyncMock()
            mock_tools = [MagicMock(name="howtocook_tool"), MagicMock(name="amap_tool")]

            mock_client_class.side_effect = [
                ConnectionError("初始连接失败"),
                mock_client,
            ]
            mock_client.get_tools.return_value = mock_tools

            # 执行启动
            await startup_mcp_client()

            # 验证结果
            status = await get_mcp_client_status()
            assert status["initialized"] is True
            assert status["tools_count"] == 2
            assert status["tools_available"] is True

            # 验证健康检查
            health = await mcp_health_check()
            assert health is True

            # 验证工具获取
            tools = await get_cached_mcp_tools()
            assert len(tools) == 2


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
