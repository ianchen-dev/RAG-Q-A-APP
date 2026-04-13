# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) Knowledge Base Q&A System** built with FastAPI, MongoDB, Redis, and LangChain. The system supports multi-format document management (PDF, Markdown, TXT), intelligent RAG Q&A with hybrid retrieval (semantic + BM25), and Agent mode with MCP (Model Context Protocol) tool integration.

**Technology Stack:**
- **Backend:** FastAPI + Uvicorn
- **Databases:** MongoDB (Beanie ODM), Redis (caching), ChromaDB (vectors)
- **AI/ML:** LangChain 0.3+, OpenAI/OneAPI/Ollama integrations
- **Package Manager:** UV
- **Testing:** Pytest with async support
- **Deployment:** Docker Compose

## Development Commands

### Environment Setup

```bash
# Install UV package manager (if not already installed)
pip install uv

# Install core dependencies only
uv sync --no-dev

# Install with dependency groups (as needed)
uv sync --group agent        # MCP tools, LangGraph (required for Agent features)
uv sync --group test         # Pytest and testing tools
uv sync --group ragas        # RAGAS evaluation framework
uv sync --group jupyter      # Jupyter notebook support
uv sync --group ai-full      # Large ML models (sentence-transformers, BERT) - ~3GB
uv sync --group semantic     # Semantic chunking (langchain-experimental)
```

### Running the Application

```bash
# Development (Windows with ProactorEventLoop for MCP compatibility)
uv run python main_dev.py

# Development with uvicorn (auto-reload)
uv run uvicorn main_dev:app --reload --host 0.0.0.0 --port 8080

# Production
uv run uvicorn main:app --host 0.0.0.0 --port 8080
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/test_file_upload_async.py -v

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Output results to file
uv run pytest > test-results/console-output.txt 2>&1
```

### Code Quality

```bash
# Format with Ruff
uv run ruff format .

# Lint with Ruff
uv run ruff check .
```

### Makefile Shortcuts

```bash
make uv          # Install agent + test dependencies
make uv-all      # Install all dependency groups (includes ragas, jupyter)
make run         # Run development server
make test        # Run tests
make test-cov    # Run tests with coverage
make build       # Build Docker image
make docker      # Run docker-compose
make CI          # Build and push image
```

### Docker Deployment

```bash
# Create shared network (if not exists)
docker network create baota_net

# Build and start services
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Architecture

### Entry Points

- **`main.py`** - Production entry point (uses `.env.prod`)
- **`main_dev.py`** - Development entry point (uses `.env.dev`, Windows ProactorEventLoop)

### Application Startup Sequence (lifespan context)

1. Load environment variables (`.env` + `.env.prod`/`.env.dev`)
2. Initialize `DatabaseManager` singleton (MongoDB + Redis connection pool)
3. Preload knowledge base cache from MongoDB to Redis
4. Create root user if not exists
5. Initialize MCP client with retry mechanism
6. Start file processing queue manager
7. Run health checks (MongoDB, Redis, OneAPI)

### Key Components

**DatabaseManager (`src/config/database_manager.py`)**
- Singleton pattern for MongoDB (Motor) and Redis (aioredis) connections
- Connection pooling, health checks, graceful shutdown
- Used by all services via `get_database_manager()`

**MCP Client Manager (`src/config/mcp_client_manager.py`)**
- Manages MCP tool connections with exponential backoff retry
- Configurable via environment variables

**Knowledge Service (`src/service/knowledgeSev.py`)**
- CRUD operations for knowledge bases
- Redis caching for knowledge base metadata
- Integrates with ChromaDB for vector storage

**Chat Service (`src/service/ChatSev.py`)**
- RAG chain with streaming output (SSE)
- Chat history persistence (MongoDB)
- Hybrid retrieval (semantic + BM25)
- Optional reranking via `src/utils/remote_rerank.py`

**File Queue Manager (`src/service/file_queue_manager.py`)**
- Async file upload queue with semaphore-based concurrency control
- Batch processing for large files

**Document Chunker (`src/components/document_chunker.py`)**
- Multiple splitting strategies: recursive, semantic, markdown, hybrid
- DocumentLoaderFactory for file type-based loader creation
- TextSplitterFactory for splitter strategy creation

### Directory Structure

```
src/
├── config/          # Configuration modules (database, logging, MCP, vector DB)
├── models/          # Beanie ODM models (MongoDB schemas)
├── router/          # FastAPI route handlers (API endpoints)
├── service/         # Business logic services
├── utils/           # Utility functions (embeddings, chunking, batch processing)
├── schema/          # Pydantic schemas for API validation
│   ├── __init__.py  # Schema package exports
│   ├── agent.py     # Agent-related schemas (QueryRequest)
│   ├── chat.py      # Chat-related schemas (LLMConfig, RerankerConfig, KnowledgeConfig, ChatConfig, ChatRequest)
│   ├── health.py    # Health check schemas (HealthCheckResponse, ConnectionStatsResponse, OneAPIHealthResponse)
│   └── knowledge.py # Knowledge base schemas (KnowledgeBaseCreate)
├── components/      # Reusable components (extracted from services via refactoring)
│   ├── __init__.py       # Component package exports
│   ├── prompt.py         # Prompt template creation component
│   ├── chat_history.py   # Chat history manager component
│   ├── chain_builder.py  # RAG/Normal chain builder component
│   └── stream_handler.py # Stream processing handler component
├── middleware/      # FastAPI middleware (CORS, logging, timing)
├── adapters/        # Adapter pattern implementations (vector databases)
│   ├── __init__.py          # Adapter package exports
│   ├── vector_db_adapter.py # Abstract base class for vector DB adapters
│   ├── chroma_adapter.py    # ChromaDB adapter implementation
│   └── milvus_adapter.py    # Milvus adapter implementation
└── factories/       # Factory pattern implementations
    ├── __init__.py          # Factory package exports
    └── vector_db_factory.py # Factory for creating vector DB adapters
```

### Environment Variables

Key environment variables (located in `.env`, `.env.dev`, or `.env.prod`):
- `MONGODB_URL` - MongoDB connection string
- `MONGO_DB_NAME` - Database name
- `ONEAPI_BASE_URL` - OneAPI gateway URL
- `ONEAPI_API_KEY` - OneAPI authentication key
- Redis connection settings
- MCP retry configuration
- Embedding model configuration

**Note:** `.env` files exist in the project root but are not visible in the codebase for privacy reasons.

## Testing Strategy

- **Unit Tests:** Fast, isolated tests marked with `@pytest.mark.unit`
- **Integration Tests:** Tests requiring external services marked with `@pytest.mark.integration`
- **Async Tests:** Uses `pytest-asyncio` with `auto` mode
- **Test Directory:** `test/` with pattern `test_*.py` or `*_test.py`
- **Output:** HTML reports, JUnit XML, coverage reports in `test-results/`

## Evaluation System

Located in `evaluation/`:
- Based on RAGAS 0.3.5+
- Config-driven evaluation via YAML files
- Supports multiple metrics: ContextRelevance, ContextPrecision, Faithfulness
- Batch processing with concurrency control
- HTML, JSON, CSV output formats

## Important Notes

1. **Windows Development:** Uses `main_dev.py` with `WindowsProactorEventLoopPolicy` for subprocess/MCP compatibility
2. **Caching Strategy:** Global module-level cache for Chroma instances with thread-safe operations
3. **Streaming:** SSE-based streaming responses for real-time chat
4. **Logging:** Separate development and production logging configurations via `src/config/logging_config.py`
5. **RAG Enhancement:** Hybrid retrieval (semantic + BM25) + optional reranking
6. **Agent Mode:** Supports web search (Tavily) and MCP tool integration
7. **Package Execution:** Always use `uv run python` for running Python commands
8. **Shell Environment:** Project uses Windows PowerShell - commands should be compatible with PowerShell
