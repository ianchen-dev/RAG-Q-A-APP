"""
文件处理组件

负责文件上传的完整处理流程，包括：
- 临时文件保存
- 文件 MD5 计算
- 同步/异步处理流程
- 临时文件清理
"""

import logging
import os
import shutil
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import UploadFile

from src.components.kb.factory import KnowledgeBaseFactory
from src.components.kb.knowledge_manager import KnowledgeManager
from src.components.kb.repository import KnowledgeBaseRepository
from src.components.kb.validator import KnowledgeBaseValidator
from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    文件处理器

    负责文件上传的完整处理流程
    """

    def __init__(
        self,
        validator: KnowledgeBaseValidator,
        kb_factory: KnowledgeBaseFactory,
        repository: KnowledgeBaseRepository,
    ):
        """
        初始化文件处理器

        Args:
            validator: 知识库验证器
            kb_factory: 知识库工厂
            repository: 知识库存储器
        """
        self.validator = validator
        self.kb_factory = kb_factory
        self.repository = repository
        self.logger = logger

    async def save_temp_file(self, file: UploadFile) -> tuple[str, str]:
        """
        保存上传的文件到临时位置

        Args:
            file: 上传的文件对象

        Returns:
            (临时文件路径, 原始文件名) 的元组
        """
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_{file.filename}"
        ) as tmp_file:
            try:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_file_path = tmp_file.name
            finally:
                file.file.close()

        self.logger.info(f"临时文件已保存: {tmp_file_path}, 文件名: {file.filename}")
        return tmp_file_path, file.filename

    async def process_file_sync(
        self,
        kb_id: str,
        file_path: str,
        file_name: str,
        file_md5: str,
        knowledge_base_doc: Optional[KnowledgeBaseModel] = None,
    ) -> dict:
        """
        同步处理已保存的文件

        Args:
            kb_id: 知识库 ID
            file_path: 文件路径
            file_name: 文件名
            file_md5: 文件 MD5 值
            knowledge_base_doc: 知识库文档对象（如果为 None 则从数据库获取）

        Returns:
            处理结果字典

        Raises:
            ValueError: 配置错误或文件已存在
            FileNotFoundError: 文件不存在
            Exception: 其他处理错误
        """
        self.logger.info(f"开始同步处理文件: {file_name} (MD5: {file_md5})")

        # 获取知识库文档
        if knowledge_base_doc is None:
            knowledge_base_doc = await self.repository.find_by_id(kb_id)
            if not knowledge_base_doc:
                raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

        # 1. 验证嵌入配置
        self.validator.validate_embedding_config(knowledge_base_doc, kb_id)

        # 2. 检查文件是否已存在
        self.validator.check_file_duplicate(
            knowledge_base_doc, kb_id, file_md5, file_name
        )

        # 3. 检查文件是否存在
        self.validator.validate_file_exists(file_path)

        # 4. 创建 KnowledgeManager 实例并处理文件
        knowledge_util = self.kb_factory.create_knowledge_manager(knowledge_base_doc)
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
        await self.repository.add_file_to_list(kb_id, file_metadata_dict)

        return {
            "message": f"文件 '{file_name}' 成功处理到知识库 '{knowledge_base_doc.title}'。",
            "knowledge_base_id": kb_id,
            "file_name": file_name,
            "file_md5": file_md5,
        }

    async def process_file_async(self, kb_id: str, file: UploadFile) -> dict:
        """
        异步处理上传的文件，快速响应并将任务加入队列

        Args:
            kb_id: 知识库 ID
            file: 上传的文件对象

        Returns:
            包含任务 ID 的响应字典

        Raises:
            FileNotFoundError: 知识库不存在
            ValueError: 配置错误
            Exception: 其他处理错误
        """
        from src.service.file_queue_manager import file_queue_manager

        # 1. 验证 kb_id 并查找 KnowledgeBase 文档
        self.validator.validate_kb_id(kb_id)
        knowledge_base_doc = await self.repository.find_by_id(kb_id)
        if not knowledge_base_doc:
            raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

        # 2. 验证嵌入配置
        self.validator.validate_embedding_config(knowledge_base_doc, kb_id)

        # 3. 快速验证和保存文件
        tmp_file_path, file_filename = await self.save_temp_file(file)

        try:
            # 4. 计算文件 MD5
            file_md5 = KnowledgeManager.get_file_md5(tmp_file_path)

            # 5. 检查文件是否已存在
            try:
                self.validator.check_file_duplicate(
                    knowledge_base_doc, kb_id, file_md5, file_filename
                )
            except ValueError:
                # 如果文件已存在，删除临时文件并重新抛出异常
                self.cleanup_temp_file(tmp_file_path)
                raise

            # 6. 将任务添加到队列
            task_id = await file_queue_manager.add_task(
                kb_id=kb_id,
                file_path=tmp_file_path,
                file_name=file_filename,
                file_md5=file_md5,
            )

            return {
                "message": f"文件 '{file_filename}' 已接收，正在后台处理",
                "task_id": task_id,
                "knowledge_base_id": kb_id,
                "file_name": file_filename,
                "file_md5": file_md5,
                "status": "queued",
            }

        except Exception:
            self.cleanup_temp_file(tmp_file_path)
            raise

    async def process_file_upload(self, kb_id: str, file: UploadFile) -> dict:
        """
        处理上传的文件，进行向量化并更新知识库记录（同步方式）

        Args:
            kb_id: 知识库 ID
            file: 上传的文件对象

        Returns:
            处理结果字典
        """
        # 1. 验证 kb_id 并查找 KnowledgeBase 文档
        self.validator.validate_kb_id(kb_id)
        knowledge_base_doc = await self.repository.find_by_id(kb_id)
        if not knowledge_base_doc:
            raise FileNotFoundError(f"知识库 ID 未找到: {kb_id}")

        # 2. 验证嵌入配置
        self.validator.validate_embedding_config(knowledge_base_doc, kb_id)

        # 3. 将上传的文件保存到临时位置
        tmp_file_path, file_filename = await self.save_temp_file(file)

        try:
            # 4. 计算文件 MD5
            file_md5 = KnowledgeManager.get_file_md5(tmp_file_path)

            # 5. 检查文件是否已存在
            self.validator.check_file_duplicate(
                knowledge_base_doc, kb_id, file_md5, file_filename
            )

            # 6. 创建 Knowledge 实例并处理文件
            knowledge_util = self.kb_factory.create_knowledge_instance(
                knowledge_base_doc
            )
            await knowledge_util.add_file_to_knowledge_base(
                kb_id=kb_id,
                file_path=tmp_file_path,
                file_name=file_filename,
                file_md5=file_md5,
            )

            # 7. 更新 MongoDB 中的 KnowledgeBase 文档
            file_metadata_dict = {
                "file_md5": file_md5,
                "file_name": file_filename,
                "upload_time": datetime.now(),
            }
            await self.repository.add_file_to_list(kb_id, file_metadata_dict)

            return {
                "message": f"文件 '{file_filename}' 成功上传并处理到知识库 '{knowledge_base_doc.title}'。",
                "knowledge_base_id": kb_id,
                "file_name": file_filename,
                "file_md5": file_md5,
            }

        except FileNotFoundError:
            self.logger.error("处理文件时未找到文件或路径")
            raise
        except ValueError:
            self.logger.error("处理文件时发生值错误")
            raise
        except Exception as e:
            self.logger.error(
                f"处理文件 {file_filename} 时发生未知错误: {e}", exc_info=True
            )
            raise
        finally:
            self.cleanup_temp_file(tmp_file_path)

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        清理临时文件

        Args:
            file_path: 要删除的临时文件路径
        """
        if os.path.exists(file_path):
            self.logger.debug(f"删除临时文件: {file_path}")
            os.remove(file_path)
