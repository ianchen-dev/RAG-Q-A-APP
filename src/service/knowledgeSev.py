import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Optional

from bson import ObjectId
from fastapi import HTTPException, UploadFile
from langchain_core.tools import tool

from src.models.knowledgeBase import (
    KnowledgeBase as KnowledgeBaseModel,
)
from src.utils.embedding import get_embedding
from src.utils.Knowledge import Knowledge

chroma_dir = "chroma/"
logger = logging.getLogger(__name__)

# Constants for batch processing
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CONCURRENT_BATCHES = 5


def _validate_embedding_config(kb_doc: KnowledgeBaseModel, kb_id: str) -> None:
    """Validate that a knowledge base has a complete embedding configuration.

    Args:
        kb_doc: Knowledge base document to validate
        kb_id: Knowledge base ID for error messages

    Raises:
        ValueError: If embedding config is missing or incomplete
    """
    if not kb_doc.embedding_config:
        raise ValueError(f"知识库 {kb_id} 缺少嵌入配置 (embedding_config)。")
    if (
        not kb_doc.embedding_config.embedding_model
        or not kb_doc.embedding_config.embedding_supplier
    ):
        raise ValueError(f"知识库 {kb_id} 的嵌入配置不完整。")


def _check_file_duplicate(
    kb_doc: KnowledgeBaseModel, kb_id: str, file_md5: str, file_name: str
) -> None:
    """Check if a file with the given MD5 already exists in the knowledge base.

    Args:
        kb_doc: Knowledge base document
        kb_id: Knowledge base ID
        file_md5: MD5 hash of the file
        file_name: Name of the file

    Raises:
        ValueError: If file already exists
    """
    if kb_doc.filesList:
        for existing_file in kb_doc.filesList:
            if existing_file.get("file_md5") == file_md5:
                logger.warning(
                    f"文件 (MD5: {file_md5}) 已存在于知识库 {kb_id}，跳过处理。"
                )
                raise ValueError(
                    f"文件 '{file_name}' (MD5: {file_md5}) 已存在于此知识库。"
                )


def _create_knowledge_instance(
    kb_doc: KnowledgeBaseModel,
    batch_size: Optional[int] = None,
    max_concurrent_batches: Optional[int] = None,
) -> Knowledge:
    """Create a Knowledge utility instance with the given knowledge base's embedding config.

    Args:
        kb_doc: Knowledge base document containing embedding config
        batch_size: Batch size for embedding (uses default if None)
        max_concurrent_batches: Max concurrent batches (uses default if None)

    Returns:
        Configured Knowledge instance
    """
    config = kb_doc.embedding_config
    logger.info(
        f"使用知识库的嵌入配置: supplier='{config.embedding_supplier}', model='{config.embedding_model}'"
    )

    _embedding = get_embedding(
        config.embedding_supplier,
        config.embedding_model,
        config.embedding_apikey,
    )

    return Knowledge(
        _embeddings=_embedding,
        vector_db_type=None,
        splitter="hybrid",
        use_bm25=False,
        use_reranker=False,
        batch_size=batch_size or DEFAULT_BATCH_SIZE,
        max_concurrent_batches=max_concurrent_batches or DEFAULT_MAX_CONCURRENT_BATCHES,
    )


async def create_knowledge(knowledge_base_data, current_user) -> KnowledgeBaseModel:
    """创建新的知识库记录"""
    new_knowledge_base = KnowledgeBaseModel(
        title=knowledge_base_data.title,
        tag=knowledge_base_data.tag,
        description=knowledge_base_data.description,
        creator=current_user.username,
        embedding_config=knowledge_base_data.embedding_config,
        filesList=[],
    )
    await new_knowledge_base.insert()
    return new_knowledge_base


async def process_uploaded_file(
    kb_id: str,
    file: UploadFile,
) -> dict:
    """处理上传的文件，进行向量化并更新知识库记录"""

    # 1. 验证 kb_id 并查找 KnowledgeBase 文档
    if not ObjectId.is_valid(kb_id):
        raise FileNotFoundError(f"无效的知识库 ID 格式: {kb_id}")
    knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(kb_id))
    if not knowledge_base_doc:
        raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

    # 2. 验证嵌入配置
    _validate_embedding_config(knowledge_base_doc, kb_id)

    # 3. 将上传的文件保存到临时位置
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    ) as tmp_file:
        try:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        finally:
            file.file.close()

    print(f"临时文件已保存: {tmp_file_path}, 文件名: {file.filename}")

    try:
        # 4. 计算文件 MD5
        file_md5 = Knowledge.get_file_md5(tmp_file_path)

        # 5. 检查文件是否已存在
        _check_file_duplicate(knowledge_base_doc, kb_id, file_md5, file.filename)

        # 6. 创建 Knowledge 实例并处理文件
        knowledge_util = _create_knowledge_instance(knowledge_base_doc)
        await knowledge_util.add_file_to_knowledge_base(
            kb_id=kb_id,
            file_path=tmp_file_path,
            file_name=file.filename,
            file_md5=file_md5,
        )

        # 7. 更新 MongoDB 中的 KnowledgeBase 文档
        file_metadata_dict = {
            "file_md5": file_md5,
            "file_name": file.filename,
            "upload_time": datetime.now(),
        }
        await knowledge_base_doc.update({"$push": {"filesList": file_metadata_dict}})
        logger.info(
            f"文件 {file.filename} (MD5: {file_md5}) 元数据已添加到 MongoDB 知识库 {kb_id}。"
        )

        return {
            "message": f"文件 '{file.filename}' 成功上传并处理到知识库 '{knowledge_base_doc.title}'。",
            "knowledge_base_id": kb_id,
            "file_name": file.filename,
            "file_md5": file_md5,
        }

    except FileNotFoundError:
        logger.error(f"处理文件时未找到文件或路径")
        raise
    except ValueError:
        logger.error(f"处理文件时发生值错误")
        raise
    except Exception as e:
        logger.error(f"处理文件 {file.filename} 时发生未知错误: {e}", exc_info=True)
        raise
    finally:
        if "tmp_file_path" in locals() and os.path.exists(tmp_file_path):
            logger.debug(f"删除临时文件: {tmp_file_path}")
            os.remove(tmp_file_path)


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
    logger.info(f"开始同步处理文件: {file_name} (MD5: {file_md5})")

    # 1. 验证嵌入配置
    _validate_embedding_config(knowledge_base_doc, kb_id)

    # 2. 检查文件是否已存在
    _check_file_duplicate(knowledge_base_doc, kb_id, file_md5, file_name)

    # 3. 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 4. 创建 Knowledge 实例并处理文件
    knowledge_util = _create_knowledge_instance(knowledge_base_doc)
    await knowledge_util.add_file_to_knowledge_base(
        kb_id=kb_id,
        file_path=file_path,
        file_name=file_name,
        file_md5=file_md5,
    )

    # 5. 更新 MongoDB 中的 KnowledgeBase 文档
    file_metadata_dict = {
        "file_md5": file_md5,
        "file_name": file_name,
        "upload_time": datetime.now(),
    }
    await knowledge_base_doc.update({"$push": {"filesList": file_metadata_dict}})
    logger.info(
        f"文件 {file_name} (MD5: {file_md5}) 元数据已添加到 MongoDB 知识库 {kb_id}。"
    )

    return {
        "message": f"文件 '{file_name}' 成功处理到知识库 '{knowledge_base_doc.title}'。",
        "knowledge_base_id": kb_id,
        "file_name": file_name,
        "file_md5": file_md5,
    }


async def process_uploaded_file_async(
    kb_id: str,
    file: UploadFile,
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
    from src.service.file_queue_manager import file_queue_manager

    # 1. 验证 kb_id 并查找 KnowledgeBase 文档
    if not ObjectId.is_valid(kb_id):
        raise FileNotFoundError(f"无效的知识库 ID 格式: {kb_id}")
    knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(kb_id))
    if not knowledge_base_doc:
        raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

    # 2. 验证嵌入配置
    _validate_embedding_config(knowledge_base_doc, kb_id)

    # 3. 快速验证和保存文件
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f"_{file.filename}"
    ) as tmp_file:
        try:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        finally:
            file.file.close()

    logger.info(f"临时文件已保存: {tmp_file_path}, 文件名: {file.filename}")

    try:
        # 4. 计算文件 MD5
        file_md5 = Knowledge.get_file_md5(tmp_file_path)

        # 5. 检查文件是否已存在
        try:
            _check_file_duplicate(knowledge_base_doc, kb_id, file_md5, file.filename)
        except ValueError:
            # 如果文件已存在，删除临时文件并重新抛出异常
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            raise

        # 6. 将任务添加到队列
        task_id = await file_queue_manager.add_task(
            kb_id=kb_id,
            file_path=tmp_file_path,
            file_name=file.filename,
            file_md5=file_md5,
        )

        return {
            "message": f"文件 '{file.filename}' 已接收，正在后台处理",
            "task_id": task_id,
            "knowledge_base_id": kb_id,
            "file_name": file.filename,
            "file_md5": file_md5,
            "status": "queued",
        }

    except Exception:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        raise


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


async def get_knowledge_list():
    """获取所有知识库列表，直接从 MongoDB 查询"""
    db_knowledge_list = await KnowledgeBaseModel.find_all().to_list()
    logger.info(f"从 MongoDB 加载了 {len(db_knowledge_list)} 个知识库。")

    return db_knowledge_list


async def delete_knowledge_base(kb_id: str) -> None:
    """删除指定的知识库及其关联的 Chroma 数据"""
    if not ObjectId.is_valid(kb_id):
        raise FileNotFoundError(f"无效的知识库 ID 格式: {kb_id}")

    # 1. 删除 MongoDB 记录
    knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(kb_id))
    if not knowledge_base_doc:
        raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

    delete_result = await knowledge_base_doc.delete()
    if not delete_result:
        logger.warning(f"MongoDB 记录 '{kb_id}' 可能未删除成功。")
        raise HTTPException(status_code=500, detail=f"删除 MongoDB 记录 {kb_id} 失败。")

    logger.info(f"MongoDB 记录 '{kb_id}' 删除成功。")

    # 2. 删除关联的向量数据库集合
    kb_id_str = str(kb_id)
    try:
        config = knowledge_base_doc.embedding_config
        if config and config.embedding_supplier and config.embedding_model:
            knowledge_util = _create_knowledge_instance(knowledge_base_doc)
            collection_deleted = await knowledge_util.delete_collection(kb_id_str)
            if collection_deleted:
                logger.info(f"向量数据库集合 '{kb_id_str}' 删除成功。")
            else:
                logger.warning(f"向量数据库集合 '{kb_id_str}' 删除失败或不存在。")
            await knowledge_util.close()
        else:
            logger.warning(f"知识库 {kb_id} 缺少嵌入配置，跳过向量数据库集合删除。")
    except Exception as e:
        logger.error(f"删除向量数据库集合 '{kb_id_str}' 时出错: {e}")


async def delete_file_from_knowledge_base(kb_id: str, file_md5: str) -> dict:
    """从指定知识库中删除特定文件（基于MD5）"""
    logger.info(f"尝试从知识库 {kb_id} 删除文件 MD5: {file_md5}")

    # 1. 验证 kb_id
    if not ObjectId.is_valid(kb_id):
        raise HTTPException(status_code=400, detail=f"无效的知识库 ID 格式: {kb_id}")

    # 2. 查找知识库文档
    knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(kb_id))
    if not knowledge_base_doc:
        raise HTTPException(status_code=404, detail=f"知识库 ID 未找到: {kb_id}")

    # 3. 更新 MongoDB: 从 filesList 移除文件信息
    logger.info(f"从 MongoDB 知识库 {kb_id} 的 filesList 中移除 MD5: {file_md5}")
    await knowledge_base_doc.update({"$pull": {"filesList": {"file_md5": file_md5}}})

    # 4. 删除向量数据库中的相关向量
    kb_id_str = str(kb_id)
    knowledge_util = _create_knowledge_instance(knowledge_base_doc)

    try:
        if await knowledge_util.collection_exists(kb_id_str):
            logger.info(
                f"准备从向量数据库集合 '{kb_id_str}' 删除与 MD5 {file_md5} 相关的向量..."
            )
            filter_dict = {"source_file_md5": file_md5}
            deletion_success = await knowledge_util.delete_documents_by_filter(
                kb_id_str, filter_dict
            )

            if deletion_success:
                logger.info(
                    f"向量数据库集合 '{kb_id_str}' 中与 MD5 {file_md5} 相关的向量已删除。"
                )
            else:
                logger.warning(
                    f"从向量数据库集合 '{kb_id_str}' 删除 MD5 {file_md5} 的向量时未成功。"
                )
        else:
            logger.info(f"向量数据库集合 '{kb_id_str}' 不存在，无需删除向量。")
    finally:
        await knowledge_util.close()

    return {"message": f"文件 MD5 {file_md5} 已成功从知识库 {kb_id} 删除。"}
