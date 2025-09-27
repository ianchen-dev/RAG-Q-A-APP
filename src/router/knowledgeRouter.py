from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

import src.service.knowledgeSev as knowledgeSev

# 导入 EmbeddingConfig 以便在 KnowledgeBaseCreate 中使用
from src.models.knowledgeBase import EmbeddingConfig
from src.service.userSev import get_current_user

knowledgeRouter = APIRouter()


# 更新 KnowledgeBaseCreate 模型以包含 EmbeddingConfig
class KnowledgeBaseCreate(BaseModel):
    title: str
    tag: Optional[list[str]] = None
    description: Optional[str] = None
    embedding_config: EmbeddingConfig  # 直接使用 EmbeddingConfig 模型


# 创建知识库
@knowledgeRouter.post("/", summary="创建知识库")
async def create_knowledge(
    knowledge_base: KnowledgeBaseCreate, current_user=Depends(get_current_user)
):
    try:
        # 服务层将接收包含 embedding_config 的 knowledge_base 对象
        new_kb = await knowledgeSev.create_knowledge(
            knowledge_base_data=knowledge_base, current_user=current_user
        )
        return new_kb
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {e}")


@knowledgeRouter.post("/{kb_id}/files/", summary="上传文件到知识库")
async def upload_file_to_knowledge_base(
    kb_id: str,
    file: UploadFile = File(...),
    # embedding_supplier: str = Form(...),
    # embedding_model: str = Form(...),
    # embedding_api_key: Optional[str] = Form(None),
):
    """
    上传单个文件到指定的知识库 (kb_id)。
    文件通过 multipart/form-data 上传。
    Embedding 相关配置将从知识库记录中获取。
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
    try:
        # 使用新的异步处理函数，快速响应并将任务加入队列
        result = await knowledgeSev.process_uploaded_file_async(
            kb_id=kb_id,
            file=file,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"上传文件到知识库 {kb_id} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理文件上传失败: {e}")

# 2025.09.20 for 后台异步上传文件功能
@knowledgeRouter.get("/tasks/{task_id}", summary="查询文件处理任务状态")
async def get_task_status(task_id: str):
    """
    查询文件处理任务的状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    try:
        from src.service.file_queue_manager import file_queue_manager
        
        # 获取任务状态
        task_status = await file_queue_manager.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在或已过期")
        
        return {
            "task_id": task_id,
            "status": task_status.get("status", "unknown"),
            "file_name": task_status.get("file_name"),
            "kb_id": task_status.get("kb_id"),
            "created_at": task_status.get("created_at"),
            "retry_count": task_status.get("retry_count", 0),
            "error_message": task_status.get("error_message")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"查询任务状态 {task_id} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"查询任务状态失败: {e}")


@knowledgeRouter.get("/queue/status", summary="查询文件处理队列状态")
async def get_queue_status():
    """
    查询文件处理队列的状态信息
    
    Returns:
        队列状态信息
    """
    try:
        from src.service.file_queue_manager import file_queue_manager
        
        # 获取队列状态
        queue_status = await file_queue_manager.get_queue_status()
        
        return {
            "queue_size": queue_status.get("queue_size", 0),
            "max_queue_size": queue_status.get("max_queue_size", 100),
            "workers_count": queue_status.get("workers_count", 0),
            "is_running": queue_status.get("is_running", False)
        }
        
    except Exception as e:
        print(f"查询队列状态时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"查询队列状态失败: {e}")


# 获取知识库列表
@knowledgeRouter.get("/", summary="获取知识库列表")
async def get_knowledge_list():
    knowledge_list = await knowledgeSev.get_knowledge_list()
    return knowledge_list


# 删除知识库
@knowledgeRouter.delete("/{kb_id}", summary="删除知识库")
async def delete_knowledge_base(kb_id: str):
    try:
        await knowledgeSev.delete_knowledge_base(kb_id)
        return {
            "message": f"Knowledge base '{kb_id}' and associated data deleted successfully."
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {e}")


# 从知识库删除指定文件
@knowledgeRouter.delete("/{kb_id}/files/{file_md5}", summary="从知识库删除指定文件")
async def delete_file_from_knowledge_base(kb_id: str, file_md5: str):
    """根据文件 MD5 从指定知识库中删除文件及其向量数据。"""
    try:
        result = await knowledgeSev.delete_file_from_knowledge_base(kb_id, file_md5)
        return result
    except HTTPException as http_exc:
        # 直接重新抛出 HTTPException，保持状态码和详情
        raise http_exc
    except FileNotFoundError as e:
        # 这个可能是 service 层没有捕获到的特定 FileNotFoundError
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        # 这个可能是 service 层没有捕获到的特定 ValueError
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 捕获 service 层可能抛出的其他 500 错误或意外错误
        print(f"删除知识库 {kb_id} 中的文件 {file_md5} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"删除文件失败: {e}")
