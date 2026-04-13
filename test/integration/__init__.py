"""
Integration tests for the application.

Integration tests require external services such as:
- External APIs (e.g., SiliconFlow Rerank API)
- Database connections (MongoDB, Redis, ChromaDB)
- Network connectivity

These tests are slower and may require specific environment setup.

Run integration tests only:
    uv run pytest test/integration/ -v

Run integration tests with coverage:
    uv run pytest test/integration/ --cov=src --cov-report=html
"""
