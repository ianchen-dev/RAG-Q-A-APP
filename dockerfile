# 使用多阶段构建
FROM python:3.10-slim-bookworm AS builder

# 设置工作目录
WORKDIR /app

# 安装 UV
RUN pip install --no-cache-dir uv

# 首先只复制依赖相关文件
COPY pyproject.toml uv.lock ./

# 导出依赖到requirements格式，然后使用uv pip安装
RUN uv export --no-dev --format requirements-txt > requirements.txt
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 安装 Node.js 和 npm (包含 npx)---MCP服务
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
