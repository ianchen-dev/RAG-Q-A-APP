# 服务重构解耦指南

## 概述

本指南基于 `ChatSev.py` 的重构经验，提供了一套标准化的服务文件解耦方法。遵循此指南可以将复杂的服务类拆分为职责单一、可测试、可维护的组件。

## 重构原则

### 1. 单一职责原则 (Single Responsibility Principle)
每个类/组件应该只有一个引起它变化的原因。识别服务中不同的职责领域，将它们分离到独立的组件中。

### 2. 依赖注入 (Dependency Injection)
通过构造函数注入依赖，而不是在组件内部创建它们。这使得组件可测试且松耦合。

### 3. 委托模式 (Delegation Pattern)
服务类作为协调者，将具体实现委托给专门的组件，保持自身的轻量和简洁。

## 重构步骤

### 步骤 1: 识别职责边界

分析服务类，识别以下类型的职责：

#### 常见职责类型
- **配置管理**: 环境变量读取、配置初始化
- **数据持久化**: 数据库操作、缓存管理
- **业务逻辑构建**: 复杂对象构建、链构建、策略选择
- **数据处理**: 序列化、反序列化、格式转换
- **外部集成**: 第三方 API 调用、消息队列
- **流式处理**: 流数据处理、分块处理
- **模板管理**: Prompt 模板、消息模板

#### ChatSev.py 职责分析示例

```python
# 原始 ChatSev.py 包含的职责：
# 1. Prompt 模板创建 → prompt.py
# 2. 聊天历史管理 → chat_history.py
# 3. RAG 链构建 → chain_builder.py
# 4. 流式输出处理 → stream_handler.py
# 5. 核心聊天协调 → ChatSev (保留)
```

### 步骤 2: 创建目录结构

 **项目分层结构**
```
src/
├── components/          # 可复用组件（从服务重构提取）
│   ├── __init__.py
│   ├── prompt.py        # Prompt 模板创建组件
│   ├── chat_history.py  # 聊天历史管理组件
│   ├── chain_builder.py # 链构建组件
│   └── stream_handler.py# 流处理组件
├── adapters/            # 适配器模式实现（向量数据库）
│   ├── __init__.py
│   ├── vector_db_adapter.py  # 向量数据库适配器抽象基类
│   ├── chroma_adapter.py      # ChromaDB 适配器
│   └── milvus_adapter.py      # Milvus 适配器
├── factories/            # 工厂模式实现
│   ├── __init__.py
│   └── vector_db_factory.py   # 向量数据库工厂
├── config/              # 配置模块
│   ├── database_manager.py    # MongoDB + Redis 连接管理器
│   ├── logging_config.py      # 日志配置
│   ├── mcp_client_manager.py  # MCP 客户端管理器
│   └── vector_db_config.py    # 向量数据库配置
├── models/              # Beanie ODM 模型（MongoDB schemas）
│   ├── knowledgeBase.py       # 知识库模型
│   ├── chat_history.py        # 聊天历史模型
│   ├── user.py                # 用户模型
│   ├── assistant.py           # 助手模型
│   └── session.py             # 会话模型
├── router/              # FastAPI 路由处理器
│   ├── chatRouter.py          # 聊天接口
│   ├── knowledgeRouter.py     # 知识库接口
│   ├── agentRouter.py         # Agent 接口
│   ├── assistantRouter.py     # 助手接口
│   ├── sessionRouter.py       # 会话接口
│   └── healthRouter.py        # 健康检查接口
├── schema/              # Pydantic 数据验证模型
│   ├── __init__.py            # Schema 包导出
│   ├── agent.py               # Agent 相关 schemas
│   ├── chat.py                # 聊天相关 schemas
│   ├── health.py              # 健康检查 schemas
│   └── knowledge.py           # 知识库相关 schemas
├── utils/               # 工具函数
│   ├── DocumentChunker.py     # 文档分块
│   ├── embedding.py           # Embedding 工具
│   ├── llm_modle.py           # LLM 工具
│   ├── batch_processor.py     # 批次处理器
│   ├── remote_rerank.py       # 远程重排序
│   └── rag_tools.py           # RAG 工具
├── middleware/          # FastAPI 中间件
│   ├── reqInfo.py             # 请求信息中间件
│   └── resTime.py             # 响应时间中间件
└── service/             # 业务逻辑服务
    └── [service].py
```


### 步骤 3: 定义组件接口

为每个组件定义清晰的接口：

```python
# 好的组件接口示例
class StreamHandler:
    """处理流式响应块的组件"""

    def __init__(self, config: HandlerConfig):
        """通过构造函数注入配置"""
        self.config = config

    def process_chunk(self, chunk: Any) -> ProcessedResult:
        """单一职责：处理数据块"""
        # 实现细节
        pass
```

### 步骤 4: 重构服务类

将服务类重构为协调者：

```python
class ChatSev:
    """聊天服务协调者"""

    def __init__(self, knowledge: Knowledge, prompt: str | None = None):
        # 初始化配置
        self.config = self._load_config()

        # 创建组件（依赖注入）
        self.chat_history_manager = ChatHistoryManager(
            mongo_connection_string=self.config.mongo_url,
            mongo_database_name=self.config.db_name,
            mongo_collection_name=self.config.collection_name,
        )

        self.chain_builder = ChainBuilder(
            knowledge=knowledge,
            knowledge_prompt=self.knowledge_prompt,
            normal_prompt=self.normal_prompt,
        )

        self.stream_handler = StreamHandler()

    def get_session_chat_history(self, session_id: str):
        """委托给专门组件"""
        return self.chat_history_manager.get_session_chat_history(session_id)

    def _create_fallback_chain(self, chat, display_name: str):
        """委托给专门组件"""
        return self.chain_builder.create_fallback_chain(chat, display_name)
```

## 组件设计模式

### 模式 1: 配置组件

用于管理配置和环境变量：

```python
# src/components/config.py
class DatabaseConfig:
    """数据库配置管理"""

    def __init__(self):
        self.connection_string = os.getenv("DATABASE_URL")
        self.database_name = os.getenv("DATABASE_NAME")
        self._validate()

    def _validate(self):
        """验证配置完整性"""
        if not self.connection_string:
            raise ValueError("DATABASE_URL is required")
```

### 模式 2: 构建器组件

用于创建复杂对象或链：

```python
# src/components/builder.py
class ChainBuilder:
    """对话链构建器"""

    def __init__(self, knowledge, prompt_template):
        self.knowledge = knowledge
        self.prompt_template = prompt_template

    async def build_chain(self, config: ChainConfig) -> RunnableChain:
        """根据配置构建链"""
        # 复杂的构建逻辑
        pass

    def _build_fallback_chain(self, error_context: str) -> RunnableChain:
        """构建备用链"""
        pass
```

### 模式 3: 处理器组件

用于处理特定类型的数据：

```python
# src/components/processor.py
class DataProcessor:
    """数据处理器"""

    def __init__(self, serializer=None):
        self.serializer = serializer or self._default_serializer

    def process(self, data: Any) -> ProcessedResult:
        """处理数据"""
        # 处理逻辑
        pass

    def _default_serializer(self, obj: Any) -> Dict:
        """默认序列化方法"""
        pass
```

### 模式 4: 管理器组件

用于管理状态和生命周期：

```python
# src/components/manager.py
class SessionManager:
    """会话管理器"""

    def __init__(self, repository: SessionRepository):
        self.repository = repository

    def get_session(self, session_id: str) -> Session:
        """获取会话"""
        return self.repository.find_by_id(session_id)

    def create_session(self, user_id: str) -> Session:
        """创建会话"""
        pass
```

## 重构检查清单

### 组件设计检查
- [ ] 组件有清晰的单一职责
- [ ] 组件通过构造函数接收依赖
- [ ] 组件有明确的公共接口
- [ ] 组件有完整的文档字符串
- [ ] 组件不依赖具体的服务实现

### 服务类检查
- [ ] 服务类作为协调者，不包含具体实现
- [ ] 服务类通过组合使用组件
- [ ] 服务类保持轻量（建议 < 300 行）
- [ ] 服务类的方法主要是委托调用

### 代码质量检查
- [ ] 所有代码通过 linter 检查
- [ ] 导入语句正确组织
- [ ] 类型注解完整
- [ ] 错误处理恰当

### 测试检查
- [ ] 每个组件可以独立测试
- [ ] 组件可以通过 mock 依赖进行单元测试
- [ ] 服务类可以通过 mock 组件进行集成测试

## 常见重构场景

### 场景 1: 提取配置逻辑

**识别信号:**
- 服务类开头有大量环境变量读取
- 配置验证逻辑混杂在业务逻辑中

**重构方案:**
```python
# 之前
class MyService:
    def __init__(self):
        self.db_url = os.getenv("DB_URL")
        if not self.db_url:
            raise ValueError("DB_URL required")
        # ... 更多配置逻辑

# 之后
class MyService:
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig()
```

### 场景 2: 提取数据处理逻辑

**识别信号:**
- 有序列化/反序列化方法
- 有格式转换逻辑
- 有数据验证逻辑

**重构方案:**
```python
# 之前
class MyService:
    def _serialize(self, obj):
        # 20+ 行序列化逻辑
        pass

# 之后
class MyService:
    def __init__(self):
        self.data_processor = DataProcessor()

    def _serialize(self, obj):
        return self.data_processor.serialize(obj)
```

### 场景 3: 提取外部集成逻辑

**识别信号:**
- 有 API 调用逻辑
- 有重试/错误处理逻辑
- 有第三方 SDK 使用

**重构方案:**
```python
# 之前
class MyService:
    async def _call_external_api(self, data):
        # API 调用逻辑
        # 重试逻辑
        # 错误处理
        pass

# 之后
class ExternalApiAdapter:
    async def call(self, data: Request) -> Response:
        # 封装所有 API 相关逻辑
        pass

class MyService:
    def __init__(self):
        self.api_adapter = ExternalApiAdapter()
```

### 场景 4: 提取复杂构建逻辑

**识别信号:**
- 有复杂的对象创建逻辑
- 有条件性的策略选择
- 有多步骤的组装过程

**重构方案:**
```python
# 之前
class MyService:
    async def _build_chain(self, config):
        # 100+ 行构建逻辑
        pass

# 之后
class ChainBuilder:
    async def build(self, config: ChainConfig) -> Chain:
        # 封装构建逻辑
        pass

class MyService:
    def __init__(self):
        self.chain_builder = ChainBuilder()
```

## 迁移策略

### 渐进式重构

不要一次性重构整个服务，采用渐进式方法：

1. **第一阶段**: 提取配置管理
2. **第二阶段**: 提取数据处理逻辑
3. **第三阶段**: 提取业务构建逻辑
4. **第四阶段**: 提取外部集成逻辑
5. **第五阶段**: 简化服务类为协调者

### 保持向后兼容

在重构过程中保持公共接口不变：

```python
class MyService:
    # 保留原有方法作为委托
    def old_method(self, param):
        return self.new_component.process(param)
```

## 参考示例

完整的重构示例请参考：
- `src/components/prompt.py` - Prompt 模板组件
- `src/components/chat_history.py` - 历史管理组件
- `src/components/chain_builder.py` - 链构建组件
- `src/components/stream_handler.py` - 流处理组件
- `src/service/ChatSev.py` - 重构后的协调者服务

## 最佳实践

1. **命名规范**
   - 组件类名: `XxxManager`, `XxxBuilder`, `XxxHandler`, `XxxProcessor`
   - 组件文件: 与类名一致的小写下划线形式
   - 方法名: 清晰描述动作

2. **文档要求**
   - 每个组件有模块级文档字符串
   - 每个公共方法有文档字符串
   - 复杂逻辑有注释说明

3. **测试要求**
   - 组件有对应的单元测试
   - 服务类有集成测试
   - 测试覆盖主要场景

4. **代码审查要点**
   - 组件职责是否单一
   - 依赖是否正确注入
   - 接口是否清晰
   - 是否可测试

## 工具支持

使用以下工具辅助重构：
- `ruff`: 代码检查和格式化
- `pytest`: 单元测试
- `mypy`: 类型检查（可选）

## 总结

本重构指南的核心思想是：
- **识别**职责边界
- **提取**独立组件
- **注入**依赖关系
- **委托**具体实现
- **简化**服务类

遵循此指南，可以将复杂的服务类重构为：
- 更易理解的代码
- 更易测试的组件
- 更易维护的架构
- 更可复用的模块
