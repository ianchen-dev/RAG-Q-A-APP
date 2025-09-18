#!/bin/bash

# =============================================================================
# RAG项目自动化部署脚本
# 功能：一键完成Docker登录、镜像拉取、服务启动
# 作者：AI Assistant
# 版本：v1.0
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_TAG="2025090v1"
DEFAULT_REGISTRY="crpi-g0adc8mdcoka4w86.cn-shenzhen.personal.cr.aliyuncs.com"
DEFAULT_USERNAME="aliyun7228596829"
DEFAULT_IMAGE_NAME="sonetto/rag"
DEFAULT_DEPLOY_DIR="/www/wwwroot/RAG"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
RAG项目自动化部署脚本

用法: $0 [选项]

选项:
    -t, --tag TAG               镜像标签 (默认: ${DEFAULT_TAG})
    -u, --username USERNAME     Docker仓库用户名 (默认: ${DEFAULT_USERNAME})
    -r, --registry REGISTRY     Docker仓库地址 (默认: ${DEFAULT_REGISTRY})
    -i, --image IMAGE           镜像名称 (默认: ${DEFAULT_IMAGE_NAME})
    -d, --dir DIRECTORY         部署目录 (默认: ${DEFAULT_DEPLOY_DIR})
    -h, --help                  显示此帮助信息
    --no-login                  跳过Docker登录步骤
    --no-pull                   跳过镜像拉取步骤
    --no-deploy                 跳过服务部署步骤
    --force                     强制重新部署（停止现有服务）

示例:
    $0                          # 使用默认配置部署
    $0 -t 2025091v2             # 指定镜像标签
    $0 --no-login               # 跳过登录步骤
    $0 --force                  # 强制重新部署

EOF
}

# 解析命令行参数
parse_args() {
    TAG="$DEFAULT_TAG"
    USERNAME="$DEFAULT_USERNAME"
    REGISTRY="$DEFAULT_REGISTRY"
    IMAGE_NAME="$DEFAULT_IMAGE_NAME"
    DEPLOY_DIR="$DEFAULT_DEPLOY_DIR"
    SKIP_LOGIN=false
    SKIP_PULL=false
    SKIP_DEPLOY=false
    FORCE_DEPLOY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            -u|--username)
                USERNAME="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -i|--image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -d|--dir)
                DEPLOY_DIR="$2"
                shift 2
                ;;
            --no-login)
                SKIP_LOGIN=true
                shift
                ;;
            --no-pull)
                SKIP_PULL=true
                shift
                ;;
            --no-deploy)
                SKIP_DEPLOY=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 构建完整镜像地址
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
}

# 检查必要的命令是否存在
check_dependencies() {
    log_info "检查系统依赖..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+="docker"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+="docker-compose"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "缺少必要的依赖: ${missing_deps[*]}"
        log_error "请先安装所需依赖后再运行此脚本"
        exit 1
    fi
    
    log_success "系统依赖检查通过"
}

# Docker登录
docker_login() {
    if [ "$SKIP_LOGIN" = true ]; then
        log_warning "跳过Docker登录步骤"
        return 0
    fi

    log_info "登录到Docker仓库: ${REGISTRY}"
    
    if docker login --username="${USERNAME}" "${REGISTRY}"; then
        log_success "Docker登录成功"
    else
        log_error "Docker登录失败"
        exit 1
    fi
}

# 拉取Docker镜像
pull_image() {
    if [ "$SKIP_PULL" = true ]; then
        log_warning "跳过镜像拉取步骤"
        return 0
    fi

    log_info "拉取Docker镜像: ${FULL_IMAGE}"
    
    if docker pull "${FULL_IMAGE}"; then
        log_success "镜像拉取成功"
    else
        log_error "镜像拉取失败"
        exit 1
    fi
}

# 检查部署目录
check_deploy_directory() {
    if [ ! -d "$DEPLOY_DIR" ]; then
        log_error "部署目录不存在: ${DEPLOY_DIR}"
        read -p "是否创建目录? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mkdir -p "$DEPLOY_DIR"
            log_success "创建目录: ${DEPLOY_DIR}"
        else
            exit 1
        fi
    fi
    
    if [ ! -f "${DEPLOY_DIR}/docker-compose.yml" ]; then
        log_warning "未找到docker-compose.yml文件: ${DEPLOY_DIR}/docker-compose.yml"
        log_warning "请确保docker-compose.yml文件存在于部署目录中"
    fi
}

# 部署服务
deploy_service() {
    if [ "$SKIP_DEPLOY" = true ]; then
        log_warning "跳过服务部署步骤"
        return 0
    fi

    log_info "切换到部署目录: ${DEPLOY_DIR}"
    cd "$DEPLOY_DIR" || {
        log_error "无法切换到部署目录: ${DEPLOY_DIR}"
        exit 1
    }

    # 如果指定了强制部署，先停止现有服务
    if [ "$FORCE_DEPLOY" = true ]; then
        log_info "强制部署模式：停止现有服务..."
        docker-compose down || log_warning "停止服务时出现警告（可能服务未运行）"
    fi

    log_info "启动Docker Compose服务..."
    
    if docker-compose up -d; then
        log_success "服务启动成功"
        
        # 显示服务状态
        log_info "当前服务状态："
        docker-compose ps
    else
        log_error "服务启动失败"
        exit 1
    fi
}

# 显示部署摘要
show_deployment_summary() {
    echo
    echo "=========================================="
    echo "           部署完成摘要"
    echo "=========================================="
    echo "镜像地址: ${FULL_IMAGE}"
    echo "部署目录: ${DEPLOY_DIR}"
    echo "部署时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "有用的命令:"
    echo "  查看日志: cd ${DEPLOY_DIR} && docker-compose logs -f"
    echo "  停止服务: cd ${DEPLOY_DIR} && docker-compose down"
    echo "  重启服务: cd ${DEPLOY_DIR} && docker-compose restart"
    echo "=========================================="
}

# 主函数
main() {
    echo
    log_info "开始RAG项目自动化部署..."
    echo
    
    # 解析参数
    parse_args "$@"
    
    # 显示配置信息
    log_info "部署配置："
    echo "  镜像: ${FULL_IMAGE}"
    echo "  部署目录: ${DEPLOY_DIR}"
    echo "  跳过登录: ${SKIP_LOGIN}"
    echo "  跳过拉取: ${SKIP_PULL}"
    echo "  跳过部署: ${SKIP_DEPLOY}"
    echo "  强制部署: ${FORCE_DEPLOY}"
    echo
    
    # 执行部署步骤
    check_dependencies
    docker_login
    pull_image
    check_deploy_directory
    deploy_service
    
    # 显示完成信息
    show_deployment_summary
    log_success "部署完成！"
}

# 错误处理
trap 'log_error "脚本执行过程中发生错误，退出码: $?"' ERR

# 执行主函数
main "$@"
