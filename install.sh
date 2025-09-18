#!/bin/bash

# =============================================================================
# RAG项目部署脚本安装器
# 功能：一键安装部署脚本到系统
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认安装目录
DEFAULT_INSTALL_DIR="/opt/rag-deploy"
INSTALL_DIR="${1:-$DEFAULT_INSTALL_DIR}"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用root用户或sudo权限运行此脚本"
        exit 1
    fi
}

# 创建安装目录
create_install_dir() {
    log_info "创建安装目录: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    log_success "安装目录创建完成"
}

# 复制脚本文件
copy_scripts() {
    log_info "复制部署脚本..."
    
    local files=("deploy.sh" "deploy-advanced.sh" "deploy.config" "README-deploy.md")
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$INSTALL_DIR/"
            log_success "已复制: $file"
        else
            log_warning "文件不存在: $file"
        fi
    done
}

# 设置权限
set_permissions() {
    log_info "设置文件权限..."
    
    chmod 755 "$INSTALL_DIR"/*.sh 2>/dev/null || true
    chmod 644 "$INSTALL_DIR"/*.config 2>/dev/null || true
    chmod 644 "$INSTALL_DIR"/*.md 2>/dev/null || true
    
    log_success "权限设置完成"
}

# 创建系统链接
create_symlinks() {
    log_info "创建系统链接..."
    
    # 创建到 /usr/local/bin 的链接
    ln -sf "$INSTALL_DIR/deploy.sh" /usr/local/bin/rag-deploy
    ln -sf "$INSTALL_DIR/deploy-advanced.sh" /usr/local/bin/rag-deploy-advanced
    
    log_success "系统链接创建完成"
    log_info "现在您可以在任何位置使用以下命令："
    echo "  - rag-deploy"
    echo "  - rag-deploy-advanced"
}

# 创建日志目录
create_log_dir() {
    log_info "创建日志目录..."
    mkdir -p /var/log
    touch /var/log/rag-deploy.log
    chmod 644 /var/log/rag-deploy.log
    log_success "日志目录设置完成"
}

# 显示安装完成信息
show_completion_info() {
    echo
    echo "=========================================="
    echo "          安装完成"
    echo "=========================================="
    echo "安装目录: $INSTALL_DIR"
    echo "配置文件: $INSTALL_DIR/deploy.config"
    echo "使用说明: $INSTALL_DIR/README-deploy.md"
    echo
    echo "快速开始："
    echo "  1. 编辑配置: nano $INSTALL_DIR/deploy.config"
    echo "  2. 执行部署: rag-deploy"
    echo "  3. 查看日志: tail -f /var/log/rag-deploy.log"
    echo
    echo "更多功能："
    echo "  rag-deploy --help"
    echo "  rag-deploy-advanced --help"
    echo "=========================================="
}

# 主函数
main() {
    echo
    log_info "开始安装RAG部署脚本..."
    echo
    
    check_root
    create_install_dir
    copy_scripts
    set_permissions
    create_symlinks
    create_log_dir
    
    show_completion_info
    log_success "安装完成！"
}

# 执行主函数
main "$@"
