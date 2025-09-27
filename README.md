# RAG-知识库问答系统

本项目是一个基于 LangChain、FastAPI、Chroma 和 MongoDB 构建的 RAG (Retrieval-Augmented Generation) 知识库问答系统,支持 MCP 服务。
![image.png](https://gitee.com/hbchen7/blog_image_hosting/raw/master/20250430153615287.png)

## 主要功能

1. 知识库管理模块
   • 多格式支持：支持 PDF、Markdown、TXT 等常见文档格式
   • 知识库创建：用户可创建多个知识库，每个知识库独立管理
   • 文档上传：批量上传文档，支持拖拽上传
   • 向量化配置：支持配置不同的嵌入模型和参数
   • 文档删除：支持删除文档

2. RAG 问答模块
   • 基于知识库的问答：用户可选择特定知识库进行问答
   • 单文件检索：支持通过 MD5 值过滤特定文档进行问答
   • 智能检索：结合语义检索和关键词检索
   • 流式输出：实时显示答案生成过程，提升用户体验
   • 多轮对话：支持上下文相关的连续问答
   • 答案溯源：显示答案来源的文档片段

3. Agent 模式

- 联网搜索
- RAG tools
- MCP 工具集成

## 技术要点

● RAG: 基于 LangChain + ChromaDB 实现文本向量化和检索，通过结构型文本解析分片、bm25 混合检索和重排序策略综合提高召回率和精确率
● Agent: 支持自主调用 RAG 功能，支持联网搜索、MCP 工具
● 后端: Fastapi + MongoDB，采用 Redis 缓存 + asyncio 实现高可用接口
● 前端: Vue3 + Elmentplus + Sass，Nginx 反向代理；基于 SSE 的流式输出
● 部署: 基于 Docker Compose 部署 后端服务

## 运行项目

### docker 运行(推荐)

1. 确保安装了 docker 环境 ->[Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. 需要使用 docker 部署 [One API](https://github.com/songquanpeng/one-api) 作为大模型网关，在 docker-compose.yml 为每个服务添加`    - baota_net # 添加对共享外部网络的连接`,接着运行`docker compose up -d`
3. 在 fastapi 项目的.env 文件中配置你的数据库连接信息等环境变量。

4. 在项目根目录下依次运行:

```bash
docker network create baota_net
```

```bash
docker-compose up --build
```

5. 运行成功后，即可访问 API 文档： `http://localhost:8080/docs`。

### 本地运行

1. 确保安装了 Python 环境 (v3.10.6+)、安装 MongoDB 数据库 (v5.0+)
2. 需要使用 docker 部署 [One API](https://github.com/songquanpeng/one-api) 作为大模型网关，部署教程：
3. 安装 uv: `pip install uv`
4. 在项目根目录.env 文件中配置数据库连接信息、 MONGODB_URL 等环境变量。
5. 在项目根目录下依次运行:

```bash
# 安装核心依赖
uv sync --no-dev

# 安装 Agent 功能（必需， MCP 服务）
uv sync --group agent

# 可选：如果需要语义分割功能
# uv sync --group semantic
```

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8080 # 启动 FastAPI 应用
```

#### 依赖说明

项目依赖已优化分组，按需安装：

- **核心依赖**：`uv sync --no-dev` - FastAPI、LangChain、数据库等基础功能
- **Agent 功能**：`uv sync --group agent` - LangGraph、MCP 服务、联网搜索（必需）
- **AI/ML 功能**：`uv sync --group ai-full` - sentence-transformers、BERT 等重量级模型（可选）
- **语义分割**：`uv sync --group semantic` - langchain-experimental（可选）

4. 浏览器访问 API 文档： `http://localhost:8080/docs`。

# 项目博客

- [langchain 项目如何实现流式输出经验分享](https://blog.csdn.net/m0_70647377/article/details/147422163)

# 关于作者

- [Github](https://github.com/hbchen7)
- [CSDN](https://blog.csdn.net/m0_70647377?spm=1000.2115.3001.5343)
