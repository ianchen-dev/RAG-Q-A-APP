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
    """
    try:
        # 调用服务层函数时不再传递 embedding 配置
        result = await knowledgeSev.process_uploaded_file(
            kb_id=kb_id,
            file=file,
            # 注释保留
            # embedding_supplier=embedding_supplier,
            # embedding_model=embedding_model,
            # embedding_api_key=embedding_api_key,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"上传文件到知识库 {kb_id} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理文件上传失败: {e}")


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
