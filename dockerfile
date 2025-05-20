# 使用多阶段构建
FROM python:3.10-slim-bookworm as builder

# 设置工作目录
WORKDIR /app

# 首先只复制依赖相关文件
COPY requirements.txt .

# 安装依赖 # --no-cache-dir
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple -r requirements.txt

# 安装 Node.js 和 npm (包含 npx)---MCP服务
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
