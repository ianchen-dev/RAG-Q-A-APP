# 测试目录结构说明

本项目的测试分为**单元测试**和**集成测试**两大类，遵循清晰的目录组织结构。

## 目录结构

```
test/
├── unit/                           # 单元测试（快速、隔离、使用 mock）
│   ├── __init__.py
│   └── src/                        # 与 src/ 目录结构对应
│       ├── config/                 # 配置模块测试
│       │   └── test_mcp_retry_mechanism.py
│       ├── service/                # 服务层测试
│       │   └── test_file_upload_async.py
│       ├── utils/                  # 工具模块测试
│       │   └── test_semaphore_batch_processing.py
│       ├── components/             # 组件测试
│       │   └── test_llm_provider.py
│       ├── adapters/               # 适配器测试
│       └── factories/              # 工厂测试
│
├── integration/                    # 集成测试（需要外部服务）
│   ├── __init__.py
│   ├── utils/                      # 工具模块集成测试
│   │   └── test_remote_rerank_integration.py
│   ├── components/                 # 组件集成测试
│   ├── service/                    # 服务集成测试
│   └── config/                     # 配置集成测试
│
└── evaluation/                     # 评估测试
    └── test_evaluation_system.py
```

## 测试类型说明

### 单元测试 (Unit Tests)
- **位置**: `test/unit/`
- **特点**:
  - 快速执行（毫秒级）
  - 完全隔离，使用 mock 模拟依赖
  - 不依赖外部服务（数据库、API 等）
  - 测试单个函数或类的行为
- **标记**: `@pytest.mark.unit`
- **运行**: `uv run pytest test/unit/ -v`

### 集成测试 (Integration Tests)
- **位置**: `test/integration/`
- **特点**:
  - 执行较慢（秒级到分钟级）
  - 需要真实的外部服务（API、数据库等）
  - 测试模块间的交互
  - 可能需要特定的环境配置
- **标记**: `@pytest.mark.integration`
- **运行**: `uv run pytest test/integration/ -v`

## 运行测试

### 运行所有测试
```bash
uv run pytest -v
```

### 只运行单元测试
```bash
uv run pytest test/unit/ -v
# 或使用标记
uv run pytest -m unit -v
```

### 只运行集成测试
```bash
uv run pytest test/integration/ -v
# 或使用标记
uv run pytest -m integration -v
```

### 运行特定测试文件
```bash
# 单元测试
uv run pytest test/unit/src/config/test_mcp_retry_mechanism.py -v

# 集成测试
uv run pytest test/integration/utils/test_remote_rerank_integration.py -v
```

### 运行特定测试类或方法
```bash
# 运行测试类
uv run pytest test/unit/src/service/test_file_upload_async.py::TestFileQueueManager -v

# 运行特定测试方法
uv run pytest test/unit/src/service/test_file_upload_async.py::TestFileQueueManager::test_add_task_to_queue -v
```

### 带覆盖率的测试
```bash
# 单元测试覆盖率
uv run pytest test/unit/ --cov=src --cov-report=html

# 集成测试覆盖率
uv run pytest test/integration/ --cov=src --cov-report=html

# 所有测试覆盖率
uv run pytest --cov=src --cov-report=html
```

## 编写测试指南

### 单元测试规范

1. **文件位置**: 测试文件应与被测试模块保持相同的目录结构
   - 被测试模块: `src/config/mcp_client_manager.py`
   - 测试文件: `test/unit/src/config/test_mcp_retry_mechanism.py`

2. **使用标记**: 在类或模块级别添加 `@pytest.mark.unit`
   ```python
   @pytest.mark.unit
   class TestMyClass:
       pass
   ```

3. **使用 mock**: 隔离外部依赖
   ```python
   from unittest.mock import AsyncMock, MagicMock, patch

   @pytest.mark.asyncio
   async def test_with_mock(self):
       mock_func = AsyncMock(return_value="success")
       result = await mock_func()
       assert result == "success"
   ```

4. **命名规范**:
   - 测试类: `Test<ClassName>`
   - 测试方法: `test_<scenario>`

### 集成测试规范

1. **文件位置**: `test/integration/<module>/test_*.py`

2. **使用标记**: 在类或模块级别添加 `@pytest.mark.integration`
   ```python
   @pytest.mark.integration
   class TestAPIIntegration:
       pass
   ```

3. **环境检查**: 使用 fixture 跳过缺少配置的测试
   ```python
   @pytest.fixture(scope="module")
   def api_key():
       key = os.getenv("API_KEY")
       if not key:
           pytest.skip("API_KEY not set")
       return key
   ```

## pytest 标记说明

本项目使用以下 pytest 标记（定义在 `pytest.ini`）：

- `unit`: 单元测试（快速、隔离）
- `integration`: 集成测试（可能需要外部服务）
- `slow`: 慢速测试（执行时间较长）
- `asyncio`: 异步测试（需要 pytest-asyncio）

## 注意事项

1. **环境变量**: 集成测试可能需要特定的环境变量，使用 `.env.dev` 配置

2. **测试隔离**: 确保测试之间相互独立，使用 `setup_method` 和 `teardown_method` 进行清理

3. **异步测试**: 所有异步测试函数必须使用 `@pytest.mark.asyncio` 装饰器

4. **Mock 使用**: 单元测试应尽可能使用 mock，避免依赖外部服务
