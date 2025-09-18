#!/bin/bash

# =============================================================================
# RAG项目高级自动化部署脚本
# 功能：支持配置文件、备份、通知等高级功能
# 作者：AI Assistant
# 版本：v2.0
# =============================================================================

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/deploy.config"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 加载配置文件
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        log_info "加载配置文件: $CONFIG_FILE"
        # 安全地加载配置文件，只允许变量赋值
        source "$CONFIG_FILE"
        log_success "配置文件加载完成"
    else
        log_warning "配置文件不存在: $CONFIG_FILE，使用默认配置"
        # 设置默认值
        TAG="${TAG:-2025090v1}"
        REGISTRY="${REGISTRY:-crpi-g0adc8mdcoka4w86.cn-shenzhen.personal.cr.aliyuncs.com}"
        USERNAME="${USERNAME:-aliyun7228596829}"
        IMAGE_NAME="${IMAGE_NAME:-sonetto/rag}"
        DEPLOY_DIR="${DEPLOY_DIR:-/www/wwwroot/RAG}"
        SKIP_LOGIN="${SKIP_LOGIN:-false}"
        SKIP_PULL="${SKIP_PULL:-false}"
        SKIP_DEPLOY="${SKIP_DEPLOY:-false}"
        FORCE_DEPLOY="${FORCE_DEPLOY:-false}"
        ENABLE_BACKUP="${ENABLE_BACKUP:-true}"
        BACKUP_DIR="${BACKUP_DIR:-/backup/rag}"
        BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
    fi
}

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    [ "${LOG_FILE:-}" ] && echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    [ "${LOG_FILE:-}" ] && echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    [ "${LOG_FILE:-}" ] && echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    [ "${LOG_FILE:-}" ] && echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# 创建备份
create_backup() {
    if [ "$ENABLE_BACKUP" != "true" ]; then
        log_info "备份功能已禁用"
        return 0
    fi

    if [ ! -d "$DEPLOY_DIR" ]; then
        log_warning "部署目录不存在，跳过备份"
        return 0
    fi

    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="${BACKUP_DIR}/rag_backup_${timestamp}"

    log_info "创建备份: $backup_path"
    
    mkdir -p "$BACKUP_DIR"
    
    if cp -r "$DEPLOY_DIR" "$backup_path"; then
        log_success "备份创建成功: $backup_path"
        
        # 清理过期备份
        cleanup_old_backups
    else
        log_error "备份创建失败"
        return 1
    fi
}

# 清理过期备份
cleanup_old_backups() {
    if [ ! -d "$BACKUP_DIR" ]; then
        return 0
    fi

    log_info "清理 ${BACKUP_RETENTION_DAYS} 天前的备份..."
    
    find "$BACKUP_DIR" -name "rag_backup_*" -type d -mtime +${BACKUP_RETENTION_DAYS} -exec rm -rf {} \; 2>/dev/null || true
    
    log_success "过期备份清理完成"
}

# 发送通知
send_notification() {
    local status="$1"
    local message="$2"
    
    if [ "$ENABLE_NOTIFICATION" != "true" ]; then
        return 0
    fi
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local notification_message="[RAG部署] ${status}: ${message} (${timestamp})"
    
    # Webhook通知
    if [ -n "${WEBHOOK_URL:-}" ]; then
        curl -X POST -H "Content-Type: application/json" \
             -d "{\"text\":\"$notification_message\"}" \
             "$WEBHOOK_URL" 2>/dev/null || log_warning "Webhook通知发送失败"
    fi
    
    # 邮件通知（需要配置mailx或sendmail）
    if [ -n "${EMAIL_RECIPIENTS:-}" ]; then
        echo "$notification_message" | mail -s "RAG部署通知" "$EMAIL_RECIPIENTS" 2>/dev/null || \
            log_warning "邮件通知发送失败"
    fi
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "Up"; then
            log_success "服务健康检查通过"
            return 0
        fi
        
        log_info "等待服务启动... (${attempt}/${max_attempts})"
        sleep 2
        ((attempt++))
    done
    
    log_error "服务健康检查失败"
    return 1
}

# 显示帮助信息
show_help() {
    cat << EOF
RAG项目高级自动化部署脚本 v2.0

用法: $0 [选项]

选项:
    -c, --config FILE           指定配置文件 (默认: ${CONFIG_FILE})
    -t, --tag TAG               镜像标签
    -h, --help                  显示此帮助信息
    --backup                    强制创建备份
    --no-backup                 跳过备份步骤
    --health-check              部署后执行健康检查
    --dry-run                   模拟运行（不执行实际操作）

配置文件选项:
    配置文件格式为shell变量格式，支持以下参数：
    - TAG: 镜像标签
    - REGISTRY: Docker仓库地址
    - USERNAME: Docker用户名
    - IMAGE_NAME: 镜像名称
    - DEPLOY_DIR: 部署目录
    - ENABLE_BACKUP: 是否启用备份
    - BACKUP_DIR: 备份目录
    - 以及更多高级选项...

示例:
    $0                          # 使用默认配置部署
    $0 -t 2025091v2             # 指定镜像标签
    $0 --health-check           # 部署后检查服务健康状态
    $0 --dry-run                # 模拟运行

EOF
}

# 模拟运行模式
dry_run() {
    echo "========== 模拟运行模式 =========="
    echo "将要执行的操作："
    echo "1. Docker登录: $REGISTRY (用户: $USERNAME)"
    echo "2. 拉取镜像: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "3. 部署目录: $DEPLOY_DIR"
    echo "4. 备份: $ENABLE_BACKUP"
    if [ "$ENABLE_BACKUP" = "true" ]; then
        echo "   备份目录: $BACKUP_DIR"
    fi
    echo "5. 启动服务: docker-compose up -d"
    echo "================================="
    echo "使用 --dry-run 参数已启用模拟模式，未执行实际操作"
}

# 解析命令行参数
parse_args() {
    DRY_RUN=false
    FORCE_BACKUP=false
    NO_BACKUP=false
    HEALTH_CHECK=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            --backup)
                FORCE_BACKUP=true
                shift
                ;;
            --no-backup)
                NO_BACKUP=true
                shift
                ;;
            --health-check)
                HEALTH_CHECK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
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
}

# 主函数
main() {
    echo
    log_info "开始RAG项目高级自动化部署..."
    echo
    
    # 解析参数
    parse_args "$@"
    
    # 加载配置
    load_config
    
    # 应用命令行覆盖
    if [ "$NO_BACKUP" = "true" ]; then
        ENABLE_BACKUP=false
    elif [ "$FORCE_BACKUP" = "true" ]; then
        ENABLE_BACKUP=true
    fi
    
    # 构建完整镜像地址
    FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
    
    # 模拟运行模式
    if [ "$DRY_RUN" = "true" ]; then
        dry_run
        return 0
    fi
    
    # 显示配置信息
    log_info "部署配置："
    echo "  镜像: ${FULL_IMAGE}"
    echo "  部署目录: ${DEPLOY_DIR}"
    echo "  备份: ${ENABLE_BACKUP}"
    echo
    
    # 创建日志目录
    if [ -n "${LOG_FILE:-}" ]; then
        mkdir -p "$(dirname "$LOG_FILE")"
    fi
    
    # 执行部署步骤
    local deployment_start_time=$(date +%s)
    
    # 备份
    if [ "$ENABLE_BACKUP" = "true" ]; then
        create_backup
    fi
    
    # 部署
    check_dependencies
    docker_login
    pull_image
    check_deploy_directory
    deploy_service
    
    # 健康检查
    if [ "$HEALTH_CHECK" = "true" ]; then
        health_check
    fi
    
    # 计算部署时间
    local deployment_end_time=$(date +%s)
    local deployment_duration=$((deployment_end_time - deployment_start_time))
    
    # 发送成功通知
    send_notification "成功" "部署完成，耗时 ${deployment_duration} 秒"
    
    # 显示完成信息
    show_deployment_summary
    log_success "部署完成！总耗时: ${deployment_duration} 秒"
}

# 从原脚本复制的函数（简化版本）
check_dependencies() {
    log_info "检查系统依赖..."
    for cmd in docker docker-compose; do
        if ! command -v $cmd &> /dev/null; then
            log_error "缺少必要的依赖: $cmd"
            exit 1
        fi
    done
    log_success "系统依赖检查通过"
}

docker_login() {
    if [ "$SKIP_LOGIN" = "true" ]; then
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

pull_image() {
    if [ "$SKIP_PULL" = "true" ]; then
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

check_deploy_directory() {
    if [ ! -d "$DEPLOY_DIR" ]; then
        log_error "部署目录不存在: ${DEPLOY_DIR}"
        exit 1
    fi
    if [ ! -f "${DEPLOY_DIR}/docker-compose.yml" ]; then
        log_warning "未找到docker-compose.yml文件"
    fi
}

deploy_service() {
    if [ "$SKIP_DEPLOY" = "true" ]; then
        log_warning "跳过服务部署步骤"
        return 0
    fi
    log_info "切换到部署目录: ${DEPLOY_DIR}"
    cd "$DEPLOY_DIR" || exit 1
    
    if [ "$FORCE_DEPLOY" = "true" ]; then
        log_info "强制部署模式：停止现有服务..."
        docker-compose down || true
    fi
    
    log_info "启动Docker Compose服务..."
    if docker-compose up -d; then
        log_success "服务启动成功"
        docker-compose ps
    else
        log_error "服务启动失败"
        exit 1
    fi
}

show_deployment_summary() {
    echo
    echo "=========================================="
    echo "           部署完成摘要"
    echo "=========================================="
    echo "镜像地址: ${FULL_IMAGE}"
    echo "部署目录: ${DEPLOY_DIR}"
    echo "部署时间: $(date '+%Y-%m-%d %H:%M:%S')"
    if [ "$ENABLE_BACKUP" = "true" ]; then
        echo "备份目录: ${BACKUP_DIR}"
    fi
    echo
    echo "有用的命令:"
    echo "  查看日志: cd ${DEPLOY_DIR} && docker-compose logs -f"
    echo "  停止服务: cd ${DEPLOY_DIR} && docker-compose down"
    echo "  重启服务: cd ${DEPLOY_DIR} && docker-compose restart"
    echo "=========================================="
}

# 错误处理
trap 'log_error "脚本执行过程中发生错误，退出码: $?"; send_notification "失败" "部署过程中发生错误"' ERR

# 执行主函数
main "$@"
