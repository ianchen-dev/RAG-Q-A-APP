# 服务重构解耦 Prompt

你是一个专业的代码重构专家，精通单一职责原则、依赖注入和设计模式。你的任务是根据现有的重构指南，对服务文件进行解耦重构。

## 背景

我们的项目已经完成了 `ChatSev.py` 的成功重构，将其拆分为：
- `src/components/prompt.py` - Prompt 模板组件
- `src/components/chat_history.py` - 历史管理组件
- `src/components/chain_builder.py` - 链构建组件
- `src/components/stream_handler.py` - 流处理组件

现在需要对其他服务文件进行类似的重构。

## 重构目标

1. **单一职责**: 每个组件只负责一个明确的职责
2. **依赖注入**: 通过构造函数注入依赖，而非内部创建
3. **委托模式**: 服务类作为协调者，委托给专门组件
4. **可测试性**: 组件可独立测试，支持 mock
5. **可维护性**: 代码清晰、文档完整

## 重构流程

### 第一步: 分析现有代码

请仔细阅读目标服务文件，分析：

1. **识别所有职责**
   - 配置管理（环境变量、配置初始化）
   - 数据持久化（数据库、缓存操作）
   - 业务逻辑构建（复杂对象创建、策略选择）
   - 数据处理（序列化、格式转换）
   - 外部集成（API 调用、第三方服务）
   - 流式处理（流数据处理、分块）
   - 其他特定职责

2. **识别依赖关系**
   - 依赖哪些外部服务
   - 依赖哪些数据库/缓存
   - 依赖哪些配置
   - 组件间的依赖关系

3. **识别可提取的部分**
   - 可以独立成组件的代码块
   - 重复的逻辑
   - 复杂的构建/处理逻辑

### 第二步: 设计组件架构

基于分析结果，设计：

1. **组件清单**
   - 需要创建哪些组件
   - 每个组件的职责
   - 组件间的依赖关系

2. **组件接口**
   - 每个组件的公共方法
   - 构造函数参数（依赖注入）
   - 返回值类型

3. **项目分层结构**
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

### 第三步: 实现组件

按顺序实现每个组件：

1. **创建组件文件**
   - 文件名: 小写下划线命名（如 `chat_history.py`）
   - 类名: 清晰描述职责（如 `ChatHistoryManager`）

2. **实现组件逻辑**
   - 从原服务类复制相关代码
   - 调整为接收依赖注入
   - 保持原有的业务逻辑

3. **添加文档**
   - 模块级文档字符串
   - 类文档字符串
   - 公共方法文档字符串

4. **更新组件导出**
   - 在 `src/components/__init__.py` 中导出新组件

### 第四步: 重构服务类

将服务类重构为协调者：

1. **添加组件导入**
   ```python
   from src.components.xxx import XxxComponent
   ```

2. **在 __init__ 中初始化组件**
   ```python
   def __init__(self, ...):
       # 配置
       self.config = self._load_config()

       # 初始化组件（依赖注入）
       self.xxx_component = XxxComponent(
           param1=self.config.xxx,
           param2=self.yyy,
       )
   ```

3. **将方法改为委托**
   ```python
   def original_method(self, ...):
       return self.xxx_component.do_something(...)
   ```

### 第五步: 验证和测试

1. **代码质量检查**
   ```bash
   uv run ruff check src/components/ src/service/
   ```

2. **导入测试**
   ```python
   from src.components import XxxComponent
   ```

3. **功能测试**（如果可能）
   - 运行相关测试
   - 验证功能完整性

## 组件设计模式

参考以下模式设计组件：

### 1. 配置组件
```python
class XxxConfig:
    """配置管理组件"""

    def __init__(self):
        self.xxx = os.getenv("XXX")
        self._validate()

    def _validate(self):
        if not self.xxx:
            raise ValueError("XXX is required")
```

### 2. 构建器组件
```python
class XxxBuilder:
    """构建器组件"""

    def __init__(self, dependency, template):
        self.dependency = dependency
        self.template = template

    def build(self, config) -> BuiltObject:
        """构建复杂对象"""
        pass

    def _build_fallback(self, error) -> BuiltObject:
        """构建备用对象"""
        pass
```

### 3. 处理器组件
```python
class XxxProcessor:
    """处理器组件"""

    def __init__(self, serializer=None):
        self.serializer = serializer or self._default_serialize

    def process(self, data) -> ProcessedResult:
        """处理数据"""
        pass

    def _default_serialize(self, obj):
        """默认序列化"""
        pass
```

### 4. 管理器组件
```python
class XxxManager:
    """管理器组件"""

    def __init__(self, repository):
        self.repository = repository

    def get(self, id) -> Entity:
        """获取实体"""
        return self.repository.find_by_id(id)

    def save(self, entity):
        """保存实体"""
        pass
```

## 注意事项

1. **保持向后兼容**
   - 不修改公共接口
   - 原有方法保持可用

2. **渐进式重构**
   - 一次提取一个组件
   - 每次重构后验证功能

3. **代码风格**
   - 遵循项目现有风格
   - 使用 ruff 格式化
   - 添加类型注解

4. **错误处理**
   - 保留原有的错误处理
   - 适当的日志记录

5. **性能考虑**
   - 避免不必要的性能损失
   - 保持原有的性能特征

## 输出要求

请按以下格式输出重构方案：

### 1. 代码分析
- 服务文件路径: `[文件路径]`
- 主要职责: `[列出识别的职责]`
- 代码行数: `[当前行数]`

### 2. 组件设计
- 需要创建的组件:
  - `[组件1名称]`: `[职责描述]`
  - `[组件2名称]`: `[职责描述]`
  - ...

### 3. 依赖关系
- 组件依赖图:
  ```
  [组件A] --> [组件B]
  [服务] --> [组件A]
  [服务] --> [组件C]
  ```

### 4. 实施计划
- 步骤 1: 创建 `[组件1]`
- 步骤 2: 创建 `[组件2]`
- 步骤 3: 重构服务类
- 步骤 4: 验证测试

### 5. 预期效果
- 代码减少: `[预计减少行数]`
- 组件数量: `[新增组件数]`
- 可测试性提升: `[描述]`

## 示例

参考 `ChatSev.py` 的重构结果：
- 原文件: ~500 行
- 重构后: 415 行（服务类）+ 629 行（4 个组件）
- 组件: prompt, chat_history, chain_builder, stream_handler

现在，请分析目标服务文件并提供重构方案。
