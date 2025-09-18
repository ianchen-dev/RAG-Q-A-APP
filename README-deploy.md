# RAG 项目自动化部署脚本使用指南

## 📋 概述

本项目提供了两个自动化部署脚本，帮助您在 Linux 服务器上一键完成 RAG 项目的 Docker 部署：

- **deploy.sh** - 基础版本，功能完整且易于使用
- **deploy-advanced.sh** - 高级版本，支持配置文件、备份、通知等高级功能

## 🚀 快速开始

### 1. 准备工作

确保您的 Linux 服务器已安装：

- Docker (版本 20.0+)
- Docker Compose (版本 1.25+)

### 2. 下载脚本

```bash
# 确保脚本有执行权限
chmod +x deploy.sh
chmod +x deploy-advanced.sh
chmod +x deploy.config
```

### 3. 基础使用

```bash
# 使用默认配置部署
./deploy.sh

# 指定镜像标签
./deploy.sh -t 2025091v2

# 跳过Docker登录（如果已登录）
./deploy.sh --no-login

# 强制重新部署
./deploy.sh --force
```

## 📖 详细功能说明

### 基础脚本 (deploy.sh)

#### 主要功能

- ✅ 自动 Docker 登录
- ✅ 镜像拉取和更新
- ✅ 服务启动和管理
- ✅ 错误处理和日志记录
- ✅ 彩色输出和进度提示

#### 命令行参数

| 参数          | 简写 | 说明             | 默认值                   |
| ------------- | ---- | ---------------- | ------------------------ |
| `--tag`       | `-t` | 镜像标签         | 2025090v1                |
| `--username`  | `-u` | Docker 用户名    | aliyun7228596829         |
| `--registry`  | `-r` | Docker 仓库地址  | crpi-g0adc8mdcoka4w86... |
| `--image`     | `-i` | 镜像名称         | sonetto/rag              |
| `--dir`       | `-d` | 部署目录         | /www/wwwroot/RAG         |
| `--no-login`  |      | 跳过 Docker 登录 | false                    |
| `--no-pull`   |      | 跳过镜像拉取     | false                    |
| `--no-deploy` |      | 跳过服务部署     | false                    |
| `--force`     |      | 强制重新部署     | false                    |
| `--help`      | `-h` | 显示帮助信息     |                          |

#### 使用示例

```bash
# 基本部署
./deploy.sh

# 指定新版本部署
./deploy.sh -t 2025091v3

# 仅拉取镜像，不部署
./deploy.sh --no-deploy

# 强制重新部署（会先停止现有服务）
./deploy.sh --force

# 自定义部署目录
./deploy.sh -d /opt/myapp/rag
```

### 高级脚本 (deploy-advanced.sh)

#### 额外功能

- 🔧 配置文件支持
- 💾 自动备份功能
- 📢 通知系统（Webhook、邮件）
- 🏥 健康检查
- 🔍 模拟运行模式
- 📊 详细的部署统计

#### 配置文件 (deploy.config)

```bash
# 编辑配置文件
nano deploy.config

# 主要配置项
TAG="2025090v1"                    # 镜像标签
REGISTRY="crpi-g0adc8mdcoka4w86..." # Docker仓库
USERNAME="aliyun7228596829"         # 用户名
DEPLOY_DIR="/www/wwwroot/RAG"       # 部署目录

# 高级选项
ENABLE_BACKUP=true                  # 启用备份
BACKUP_DIR="/backup/rag"            # 备份目录
BACKUP_RETENTION_DAYS=7             # 备份保留天数
LOG_FILE="/var/log/rag-deploy.log"  # 日志文件
```

#### 高级用法示例

```bash
# 使用配置文件部署
./deploy-advanced.sh

# 模拟运行（查看将要执行的操作）
./deploy-advanced.sh --dry-run

# 部署后执行健康检查
./deploy-advanced.sh --health-check

# 强制创建备份
./deploy-advanced.sh --backup

# 跳过备份
./deploy-advanced.sh --no-backup

# 使用自定义配置文件
./deploy-advanced.sh -c /path/to/custom.config
```

## 🔧 环境配置

### 目录结构要求

```
/www/wwwroot/RAG/           # 部署目录
├── docker-compose.yml      # Docker Compose配置文件
├── .env                    # 环境变量文件（可选）
└── data/                   # 数据目录（可选）
```

### docker-compose.yml 示例

```yaml
version: "3.8"
services:
  rag-app:
    image: crpi-g0adc8mdcoka4w86.cn-shenzhen.personal.cr.aliyuncs.com/sonetto/rag:${TAG:-2025090v1}
    container_name: rag-application
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📋 常见使用场景

### 场景 1：日常更新部署

```bash
# 1. 拉取最新镜像并部署
./deploy.sh -t latest

# 2. 查看服务状态
cd /www/wwwroot/RAG && docker-compose ps

# 3. 查看日志
docker-compose logs -f
```

### 场景 2：版本回滚

```bash
# 1. 停止当前服务
cd /www/wwwroot/RAG && docker-compose down

# 2. 部署指定版本
./deploy.sh -t 2025090v1 --force

# 3. 验证服务正常
./deploy-advanced.sh --health-check
```

### 场景 3：定期备份部署

```bash
# 创建定时任务
echo "0 2 * * * /path/to/deploy-advanced.sh --backup --no-deploy" | crontab -

# 手动备份
./deploy-advanced.sh --backup --no-deploy
```

### 场景 4：生产环境部署

```bash
# 1. 先模拟运行检查配置
./deploy-advanced.sh --dry-run

# 2. 创建备份后部署
./deploy-advanced.sh --backup --health-check

# 3. 监控部署过程
tail -f /var/log/rag-deploy.log
```

## 🚨 故障排除

### 常见问题

#### 1. Docker 登录失败

```bash
# 问题：Authentication failed
# 解决：检查用户名和密码
./deploy.sh --no-login  # 跳过登录步骤
```

#### 2. 镜像拉取失败

```bash
# 问题：Pull access denied
# 解决：确保网络连接和权限
docker login crpi-g0adc8mdcoka4w86.cn-shenzhen.personal.cr.aliyuncs.com
```

#### 3. 部署目录不存在

```bash
# 问题：Directory not found
# 解决：创建目录或修改配置
mkdir -p /www/wwwroot/RAG
./deploy.sh -d /www/wwwroot/RAG
```

#### 4. 端口冲突

```bash
# 问题：Port already in use
# 解决：停止冲突服务或修改端口
docker-compose down
./deploy.sh --force
```

### 日志查看

```bash
# 部署脚本日志
tail -f /var/log/rag-deploy.log

# Docker Compose日志
cd /www/wwwroot/RAG && docker-compose logs -f

# 系统Docker日志
journalctl -u docker -f
```

## 🔐 安全建议

1. **权限控制**

   ```bash
   # 设置脚本权限
   chmod 750 deploy*.sh
   chown root:docker deploy*.sh
   ```

2. **敏感信息保护**

   ```bash
   # 保护配置文件
   chmod 600 deploy.config

   # 使用环境变量替代硬编码密码
   export DOCKER_PASSWORD="your_password"
   ```

3. **日志安全**
   ```bash
   # 设置日志轮转
   echo "/var/log/rag-deploy.log {
       daily
       rotate 7
       compress
       missingok
   }" > /etc/logrotate.d/rag-deploy
   ```

## 📞 支持与维护

### 脚本更新

定期检查脚本更新，获取最新功能和安全修复。

### 监控建议

- 设置服务监控告警
- 定期检查备份完整性
- 监控磁盘空间使用

### 联系支持

如有问题，请提供：

- 错误日志
- 系统环境信息
- 操作步骤说明

---

_最后更新：2025-09-14_
_版本：v2.0_
