# 向量数据库架构设计文档

## 概述

本项目采用工厂模式和适配器模式重构了向量数据库架构，支持多种向量数据库（Chroma 和 Milvus），提供统一的接口和灵活的配置。

## 架构设计

### 设计模式

1. **工厂模式（Factory Pattern）**

   - `VectorDBFactory`: 创建不同类型的向量数据库适配器
   - `VectorDBManager`: 管理向量数据库实例和集合

2. **适配器模式（Adapter Pattern）**
   - `VectorDBAdapter`: 抽象基类，定义统一接口
   - `ChromaAdapter`: Chroma 数据库适配器实现
   - `MilvusAdapter`: Milvus 数据库适配器实现
   - `VectorStoreAdapter`: 向量存储适配器，兼容 Langchain 接口

### 组件结构

```
src/
├── config/
│   ├── vectorDB.config          # 向量数据库配置文件
│   └── vector_db_config.py      # 配置模型和加载器
├── adapters/
│   ├── vector_db_adapter.py     # 适配器接口定义
│   ├── chroma_adapter.py        # Chroma适配器实现
│   └── milvus_adapter.py        # Milvus适配器实现
├── factories/
│   └── vector_db_factory.py     # 工厂类和管理器
└── utils/
    └── Knowledge.py             # 重构后的Knowledge类
```

## 配置说明

### 环境变量配置

在`.env`文件或环境变量中设置：

```bash
# 默认向量数据库类型
DEFAULT_VECTOR_DB_TYPE=chroma

# Chroma 配置
CHROMA_PERSIST_DIRECTORY=chroma/
CHROMA_COLLECTION_METADATA={"hnsw:space": "cosine"}

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB_NAME=default
MILVUS_SECURE=false
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_METRIC_TYPE=L2
MILVUS_NLIST=1024
```

### 支持的数据库类型

- `chroma`: ChromaDB 向量数据库
- `milvus`: Milvus 向量数据库

## 使用示例

### 基本使用

```python
from src.utils.Knowledge import Knowledge
from src.utils.embedding import get_embedding
from src.config.vector_db_config import VectorDBType

# 获取嵌入模型
embedding = get_embedding("oneapi", "BAAI/bge-m3", "your-api-key")

# 创建Knowledge实例
knowledge = Knowledge(
    _embeddings=embedding,
    vector_db_type=VectorDBType.CHROMA,  # 指定使用Chroma
    splitter="hybrid",
    use_bm25=True,      # 启用BM25混合检索
    use_reranker=True,  # 启用重排序
    remote_rerank_config={
        "api_key": "your-rerank-api-key",
        "model": "BAAI/bge-reranker-v2-m3"
    }
)

# 添加文件到知识库
await knowledge.add_file_to_knowledge_base(
    kb_id="kb_001",
    file_path="/path/to/document.pdf",
    file_name="document.pdf",
    file_md5="file_md5_hash"
)

# 获取检索器
retriever = await knowledge.get_retriever_for_knowledge_base(
    kb_id="kb_001",
    search_k=5
)

# 检索文档
documents = await retriever.ainvoke("查询问题")
```

### 使用工厂模式

```python
from src.factories.vector_db_factory import VectorDBFactory, VectorDBManager
from src.config.vector_db_config import VectorDBType

# 创建适配器
adapter = VectorDBFactory.create_adapter(embedding, VectorDBType.MILVUS)

# 使用管理器
async with VectorDBManager(embedding, VectorDBType.CHROMA) as manager:
    # 创建集合
    await manager.create_collection("test_collection")

    # 获取集合适配器
    collection = manager.get_collection("test_collection")

    # 添加文档
    await collection.aadd_documents(documents)

    # 获取检索器
    retriever = collection.as_retriever(search_kwargs={"k": 3})
```

## 特性支持

### 多向量数据库支持

- **Chroma**: 轻量级向量数据库，适合小到中等规模应用
- **Milvus**: 高性能向量数据库，适合大规模生产环境

### 高级检索功能

1. **BM25 混合检索**: 结合密集向量和稀疏检索
2. **重排序**: 支持远程重排序 API 提升检索精度
3. **元数据过滤**: 基于文档元数据的精确过滤

### 兼容性

- 完全兼容现有的 Knowledge 类接口
- 支持 Langchain 检索器接口
- 向后兼容现有的 knowledgeSev.py 服务

## 扩展指南

### 添加新的向量数据库

1. 创建新的适配器类继承`VectorDBAdapter`
2. 实现所有抽象方法
3. 在`VectorDBFactory`中添加支持
4. 更新配置文件和枚举类型

示例：

```python
class MyVectorDBAdapter(VectorDBAdapter):
    def __init__(self, embeddings: Embeddings):
        super().__init__(embeddings)
        # 初始化代码

    async def create_collection(self, collection_name: str) -> bool:
        # 实现创建集合逻辑
        pass

    # 实现其他抽象方法...
```

### 配置新的检索策略

可以通过修改 Knowledge 类的初始化参数来配置不同的检索策略：

```python
# 纯向量检索
knowledge = Knowledge(_embeddings=embedding, use_bm25=False, use_reranker=False)

# BM25混合检索
knowledge = Knowledge(_embeddings=embedding, use_bm25=True, bm25_k=5)

# 重排序检索
knowledge = Knowledge(
    _embeddings=embedding,
    use_reranker=True,
    reranker_type="remote",
    remote_rerank_config={"api_key": "key", "model": "model"}
)
```

## 性能优化建议

1. **连接池管理**: 对于 Milvus，建议使用连接池
2. **批量操作**: 大批量文档添加时使用批处理
3. **索引优化**: 根据数据特点调整索引参数
4. **缓存策略**: 合理使用集合缓存避免重复加载

## 故障排除

### 常见问题

1. **依赖库缺失**

   - Chroma: `pip install langchain-chroma`
   - Milvus: `pip install pymilvus`

2. **配置错误**

   - 检查环境变量设置
   - 验证向量数据库服务是否运行

3. **连接失败**
   - 检查网络连接
   - 验证认证信息

### 日志调试

启用详细日志来诊断问题：

```python
import logging
logging.getLogger("src.adapters").setLevel(logging.DEBUG)
logging.getLogger("src.factories").setLevel(logging.DEBUG)
```

## 总结

新的向量数据库架构提供了：

- **灵活性**: 支持多种向量数据库
- **可扩展性**: 易于添加新的数据库支持
- **兼容性**: 保持现有接口不变
- **性能**: 优化的检索策略和资源管理
- **维护性**: 清晰的模块分离和依赖注入

这个设计为项目的未来扩展奠定了坚实的基础。
