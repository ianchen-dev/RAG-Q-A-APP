# RAG-知识库问答系统

本项目是一个基于 LangChain、FastAPI、Chroma 和 MongoDB 构建的 RAG (Retrieval-Augmented Generation) 知识库问答系统,支持 MCP 服务。
![image.png](https://gitee.com/hbchen7/blog_image_hosting/raw/master/20250430153615287.png)

## 主要功能

### RAG

- 单文件检索：支持通过元数据过滤（指定文件 MD5）实现对知识库中特定文件的检索问答。
- BM25 混合检索
- 重排序

### Agent

- 联网搜索
- MCP 服务

### LLM 对话

- 流式输出
- 助手会话管理
- 提示词自定义
- 历史消息功能

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
3. 安装 pdm: `pip install pdm`
4. 在项目根目录.env 文件中配置数据库连接信息、 MONGODB_URL 等环境变量。
5. 在项目根目录下依次运行:

```bash
pdm init # 初始化 PDM 项目，创建虚拟环境
```

```bash
pdm install # 安装项目依赖
```

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080 # 启动 FastAPI 应用
```

4. 浏览器访问 API 文档： `http://localhost:8080/docs`。

# 项目博客

- [langchain 项目如何实现流式输出经验分享](https://blog.csdn.net/m0_70647377/article/details/147422163)

# 关于作者

- ~~[个人网站]()~~:挖坑待填,敬请期待~
- [Github](https://github.com/hbchen7)
- [CSDN](https://blog.csdn.net/m0_70647377?spm=1000.2115.3001.5343)
- [BiliBili](https://space.bilibili.com/1608655290)

# 特此鸣谢

- [One API](https://github.com/songquanpeng/one-api)
- [langchain-API 文档](https://python.langchain.com/api_reference/)

# 特此鸣谢

- [One API](https://github.com/songquanpeng/one-api)
- [langchain-API 文档](https://python.langchain.com/api_reference/)
