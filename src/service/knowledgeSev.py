"""
知识库服务模块

提供知识库的 CRUD 操作和文件管理功能
重构后使用组件化架构，通过依赖注入委托给专门组件
"""

import logging
from typing import List

from fastapi import HTTPException
from langchain_core.tools import tool

from src.components import (
    FileProcessor,
    KnowledgeBaseFactory,
    KnowledgeBaseRepository,
    KnowledgeBaseValidator,
    VectorDBManager,
)
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel

logger = logging.getLogger(__name__)

# 全局组件实例（延迟初始化）
_validator: KnowledgeBaseValidator | None = None
_factory: KnowledgeBaseFactory | None = None
_repository: KnowledgeBaseRepository | None = None
_file_processor: FileProcessor | None = None
_vector_db_manager: VectorDBManager | None = None


def _get_components():
    """获取或初始化组件实例（延迟初始化）"""
    global _validator, _factory, _repository, _file_processor, _vector_db_manager

    if _validator is None:
        _validator = KnowledgeBaseValidator()
    if _factory is None:
        _factory = KnowledgeBaseFactory()
    if _repository is None:
        _repository = KnowledgeBaseRepository()
    if _file_processor is None:
        _file_processor = FileProcessor(_validator, _factory, _repository)
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager(_factory)

    return _validator, _factory, _repository, _file_processor, _vector_db_manager


# ============================================================================
# 公共 API 函数
# ============================================================================


async def create_knowledge(knowledge_base_data, current_user) -> KnowledgeBaseModel:
    """创建新的知识库记录"""
    _, factory, repository, _, _ = _get_components()

    new_knowledge_base = factory.create_knowledge_base(
        knowledge_base_data, current_user
    )
    return await repository.save(new_knowledge_base)


async def process_uploaded_file(
    kb_id: str,
    file,
) -> dict:
    """处理上传的文件，进行向量化并更新知识库记录（同步方式）"""
    _, _, _, file_processor, _ = _get_components()
    return await file_processor.process_file_upload(kb_id, file)


async def process_uploaded_file_sync(
    kb_id: str,
    file_path: str,
    file_name: str,
    file_md5: str,
    knowledge_base_doc: KnowledgeBaseModel,
) -> dict:
    """
    同步处理已保存的文件，用于异步队列处理

    Args:
        kb_id: 知识库ID
        file_path: 文件路径
        file_name: 文件名
        file_md5: 文件MD5值
        knowledge_base_doc: 知识库文档对象

    Returns:
        处理结果字典

    Raises:
        ValueError: 配置错误或文件已存在
        FileNotFoundError: 文件不存在
        Exception: 其他处理错误
    """
    _, _, _, file_processor, _ = _get_components()
    return await file_processor.process_file_sync(
        kb_id, file_path, file_name, file_md5, knowledge_base_doc
    )


async def process_uploaded_file_async(
    kb_id: str,
    file,
) -> dict:
    """
    异步处理上传的文件，快速响应并将任务加入队列

    Args:
        kb_id: 知识库ID
        file: 上传的文件对象

    Returns:
        包含任务ID的响应字典

    Raises:
        FileNotFoundError: 知识库不存在
        ValueError: 配置错误
        Exception: 其他处理错误
    """
    _, _, _, file_processor, _ = _get_components()
    return await file_processor.process_file_async(kb_id, file)


@tool
async def get_knowledge_list_tool():
    """获取所有知识库列表：
    [
    {
        "_id": "67ff248e0e67faaaae7c5303",
        "title": "RAG知识库",
        "tag": [
            "RAG知识库"
        ],
        "description": "RAG知识库",
        "creator": "user1",
        "filesList": [
            {
                "file_md5": "1d49477ffcd597016fbdcf09e95c7e41",
                "file_path": "C:\\Users\\hbche\\AppData\\Local\\Temp\\tmpv71f5vt8_RAG-QA-PRD.pdf",
                "file_name": "RAG-QA-PRD.pdf",
                "upload_time": "2025-04-26T09:23:10.054000"
            },

            {
                "file_md5": "b7c89c84a612852fe4c2e1c6c1882530",
                "file_name": "RAG.md",
                "upload_time": "2025-05-04T19:00:25.540000"
            }

        ],
        "embedding_config": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_supplier": "oneapi",
            "embedding_apikey": "sk-enlDKhEcgGKyeJPx5b8c65Dc9d9b4842A24f5223A4Fb50C3"
        },
        "create_at": "2025-04-16T11:31:26.195000"
    }
    ]
    """

    return await get_knowledge_list()


async def get_knowledge_list() -> List[KnowledgeBaseModel]:
    """获取所有知识库列表，直接从 MongoDB 查询"""
    _, _, repository, _, _ = _get_components()
    return await repository.find_all()


async def delete_knowledge_base(kb_id: str) -> None:
    """删除指定的知识库及其关联的向量数据库数据"""
    validator, _, repository, _, vector_db_manager = _get_components()

    # 验证 kb_id
    validator.validate_kb_id(kb_id)

    # 1. 删除 MongoDB 记录
    knowledge_base_doc = await repository.find_by_id(kb_id)
    if not knowledge_base_doc:
        raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

    delete_result = await repository.delete(kb_id)
    if not delete_result:
        logger.warning(f"MongoDB 记录 '{kb_id}' 可能未删除成功。")
        raise HTTPException(status_code=500, detail=f"删除 MongoDB 记录 {kb_id} 失败。")

    # 2. 删除关联的向量数据库集合
    await vector_db_manager.delete_collection(kb_id, knowledge_base_doc)


async def delete_file_from_knowledge_base(kb_id: str, file_md5: str) -> dict:
    """从指定知识库中删除特定文件（基于MD5）"""
    validator, _, repository, _, vector_db_manager = _get_components()

    logger.info(f"尝试从知识库 {kb_id} 删除文件 MD5: {file_md5}")

    # 1. 验证 kb_id
    try:
        validator.validate_kb_id(kb_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. 查找知识库文档
    knowledge_base_doc = await repository.find_by_id(kb_id)
    if not knowledge_base_doc:
        raise HTTPException(status_code=404, detail=f"知识库 ID 未找到: {kb_id}")

    # 3. 更新 MongoDB: 从 filesList 移除文件信息
    await repository.remove_file_from_list(kb_id, file_md5)

    # 4. 删除向量数据库中的相关向量
    filter_dict = {"source_file_md5": file_md5}
    await vector_db_manager.delete_vectors_by_filter(kb_id, filter_dict, knowledge_base_doc)

    return {"message": f"文件 MD5 {file_md5} 已成功从知识库 {kb_id} 删除。"}
