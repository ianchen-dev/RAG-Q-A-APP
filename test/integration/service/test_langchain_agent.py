"""
Integration tests for langchain_agent.py module.

These tests require:
1. Valid MongoDB connection (MONGODB_URL, MONGO_DB_NAME, MONGODB_COLLECTION_NAME_CHATHISTORY)
2. Valid LLM API keys (VOLCES_API_KEY, VOLCES_MODEL)
3. MCP client manager (optional, tests will skip if not available)

Run with: uv run pytest test/integration/service/test_langchain_agent.py -v
"""

import asyncio
import os
from typing import Any, AsyncGenerator

import pytest
from dotenv import load_dotenv

from src.service.langchain_agent import (
    LangChainAgent,
    get_langchain_agent,
    main_graph_execution,
)


@pytest.fixture(scope="module")
def agent_env() -> dict[str, str]:
    """
    Load environment variables for agent tests.

    Skips all tests in this module if required env vars are not available.
    """
    load_dotenv(dotenv_path=".env.dev", override=True)

    required_vars = {
        "MONGODB_URL": os.getenv("MONGODB_URL"),
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME"),
        "MONGODB_COLLECTION_NAME_CHATHISTORY": os.getenv("MONGODB_COLLECTION_NAME_CHATHISTORY"),
        "VOLCES_API_KEY": os.getenv("VOLCES_API_KEY"),
        "VOLCES_MODEL": os.getenv("VOLCES_MODEL"),
    }

    missing_vars = [k for k, v in required_vars.items() if not v]

    if missing_vars:
        pytest.skip(
            f"Required environment variables not set: {', '.join(missing_vars)}. "
            "Skipping integration tests."
        )

    return required_vars


@pytest.mark.integration
class TestLangChainAgent:
    """Integration tests for LangChainAgent class."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_env: dict[str, str]) -> None:
        """Test agent initialization with required environment variables."""
        agent = LangChainAgent()

        assert agent.mongo_connection_string is not None
        assert agent.mongo_database_name is not None
        assert agent.mongo_collection_name is not None
        assert len(agent.base_tools) >= 2  # At least knowledge tools

    @pytest.mark.asyncio
    async def test_get_session_chat_history(self, agent_env: dict[str, str]) -> None:
        """Test getting session chat history."""
        agent = LangChainAgent()
        session_id = "test_session_123"

        history = agent.get_session_chat_history(session_id)

        assert history is not None
        assert history.session_id == session_id

    @pytest.mark.asyncio
    async def test_get_all_tools(self, agent_env: dict[str, str]) -> None:
        """Test getting all available tools."""
        agent = LangChainAgent()
        tools = await agent._get_all_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 2  # At least knowledge tools

        # Check tool structure
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")

    @pytest.mark.asyncio
    async def test_stream_chat_simple_query(self, agent_env: dict[str, str]) -> None:
        """Test stream_chat with a simple query."""
        agent = LangChainAgent()
        session_id = "test_session_simple"
        user_input = "你好"

        events = []
        async for event in agent.stream_chat(user_input, session_id):
            events.append(event)

        # Should receive at least stream_end event
        assert any(e.get("type") == "stream_end" for e in events)

    @pytest.mark.asyncio
    async def test_stream_chat_with_tool_call(self, agent_env: dict[str, str]) -> None:
        """Test stream_chat with a query that triggers tool calls."""
        agent = LangChainAgent()
        session_id = "test_session_tool"
        user_input = "请帮我查询知识库列表"

        events = []
        async for event in agent.stream_chat(user_input, session_id):
            events.append(event)
            # Break after receiving some events to avoid long execution
            if len(events) > 10:
                break

        # Check for expected event types
        event_types = {e.get("type") for e in events}
        assert "stream_end" in event_types or len(events) > 0


@pytest.mark.integration
class TestLangChainAgentSingleton:
    """Integration tests for LangChain agent singleton pattern."""

    @pytest.mark.asyncio
    async def test_get_langchain_agent_singleton(self, agent_env: dict[str, str]) -> None:
        """Test that get_langchain_agent returns same instance."""
        agent1 = get_langchain_agent()
        agent2 = get_langchain_agent()

        assert agent1 is agent2


@pytest.mark.integration
class TestMainGraphExecution:
    """Integration tests for main_graph_execution function."""

    @pytest.mark.asyncio
    async def test_main_graph_execution_simple(self, agent_env: dict[str, str]) -> None:
        """Test main_graph_execution with simple query."""
        session_id = "test_main_graph_123"
        user_input = "你好"

        events = []
        async for event in main_graph_execution(user_input, session_id):
            events.append(event)
            # Limit events for faster test
            if len(events) > 5:
                break

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_main_graph_execution_event_types(self, agent_env: dict[str, str]) -> None:
        """Test main_graph_execution produces expected event types."""
        session_id = "test_main_graph_types"
        user_input = "你好"

        events = []
        async for event in main_graph_execution(user_input, session_id):
            events.append(event)
            if event.get("type") == "stream_end":
                break

        # Check event structure
        for event in events:
            assert "type" in event
            assert "data" in event or event.get("type") == "stream_end"


@pytest.mark.integration
class TestLangChainAgentErrorHandling:
    """Integration tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_missing_environment_variables(self) -> None:
        """Test that missing environment variables raise ValueError."""
        # Temporarily clear environment variables
        original_url = os.environ.get("MONGODB_URL")
        original_db = os.environ.get("MONGO_DB_NAME")
        original_collection = os.environ.get("MONGODB_COLLECTION_NAME_CHATHISTORY")

        try:
            os.environ.pop("MONGODB_URL", None)
            os.environ.pop("MONGO_DB_NAME", None)
            os.environ.pop("MONGODB_COLLECTION_NAME_CHATHISTORY", None)

            # Reload env vars
            load_dotenv(dotenv_path=".env.dev", override=True)

            with pytest.raises(ValueError, match="环境变量.*未设置"):
                LangChainAgent()

        finally:
            # Restore environment variables
            if original_url:
                os.environ["MONGODB_URL"] = original_url
            if original_db:
                os.environ["MONGO_DB_NAME"] = original_db
            if original_collection:
                os.environ["MONGODB_COLLECTION_NAME_CHATHISTORY"] = original_collection
