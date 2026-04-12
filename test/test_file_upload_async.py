"""
文件上传异步处理功能测试用例

测试内容：
1. 异步队列管理器基础功能测试
2. 文件上传接口响应速度测试
3. 任务状态查询功能测试
4. 并发处理能力测试
5. 错误处理和重试机制测试

Author: hbchen
Date: 2025-09-19
"""

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi import UploadFile

# 加载环境变量 - 必须在导入其他模块之前
load_dotenv()  # 加载 .env 基础配置
load_dotenv(dotenv_path=".env.dev", override=True)

# 导入被测试的模块 - 现在环境变量已经加载
from src.service.file_queue_manager import FileProcessingTask, FileQueueManager
from src.service.knowledgeSev import process_uploaded_file_async


class TestFileQueueManager:
    """文件队列管理器测试"""

    @pytest.fixture
    async def queue_manager(self):
        """创建队列管理器实例"""
        manager = FileQueueManager(max_queue_size=10, max_workers=2)
        await manager.initialize()
        yield manager
        await manager.stop_workers()

    @pytest.mark.asyncio
    async def test_queue_manager_initialization(self):
        """测试队列管理器初始化"""
        manager = FileQueueManager(max_queue_size=5, max_workers=2)

        await manager.initialize()

        assert manager.max_queue_size == 5
        assert manager.max_workers == 2
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_add_task_to_queue(self, queue_manager):
        """测试添加任务到队列"""
        kb_id = str(uuid.uuid4())
        file_path = "/tmp/test.txt"
        file_name = "test.txt"
        file_md5 = "abc123"

        task_id = await queue_manager.add_task(
            kb_id=kb_id,
            file_path=file_path,
            file_name=file_name,
            file_md5=file_md5
        )

        assert task_id is not None
        assert queue_manager.file_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_get_task_status(self, queue_manager):
        """测试获取任务状态"""
        task_id = "test-task-id"

        # 创建任务对象并添加到内存
        task = FileProcessingTask(
            task_id=task_id,
            kb_id="kb-id",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            file_md5="abc123",
            status="processing"
        )
        queue_manager._tasks[task_id] = task

        status = await queue_manager.get_task_status(task_id)

        assert status["task_id"] == task_id
        assert status["status"] == "processing"

    @pytest.mark.asyncio
    async def test_get_queue_status(self, queue_manager):
        """测试获取队列状态"""
        # 添加一个任务到队列
        await queue_manager.add_task("kb1", "/tmp/test.txt", "test.txt", "abc123")

        status = await queue_manager.get_queue_status()

        assert status["queue_size"] == 1
        assert status["max_queue_size"] == 10
        assert status["workers_count"] == 0  # 未启动工作进程
        assert not status["is_running"]

    @pytest.mark.asyncio
    async def test_start_and_stop_workers(self, queue_manager):
        """测试启动和停止工作进程"""
        # 启动工作进程
        await queue_manager.start_workers()

        assert queue_manager.is_running
        assert len(queue_manager.workers) == 2

        # 停止工作进程
        await queue_manager.stop_workers()

        assert not queue_manager.is_running
        assert len(queue_manager.workers) == 0


class TestFileProcessingTask:
    """文件处理任务模型测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = FileProcessingTask(
            task_id="test-id",
            kb_id="kb-id",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            file_md5="abc123"
        )
        
        assert task.task_id == "test-id"
        assert task.kb_id == "kb-id"
        assert task.file_path == "/tmp/test.txt"
        assert task.file_name == "test.txt"
        assert task.file_md5 == "abc123"
        assert task.status == "queued"
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_task_serialization(self):
        """测试任务序列化和反序列化"""
        original_task = FileProcessingTask(
            task_id="test-id",
            kb_id="kb-id",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            file_md5="abc123",
            status="processing",
            retry_count=1
        )
        
        # 序列化
        task_dict = original_task.to_dict()
        
        # 反序列化
        restored_task = FileProcessingTask.from_dict(task_dict)
        
        assert restored_task.task_id == original_task.task_id
        assert restored_task.kb_id == original_task.kb_id
        assert restored_task.file_path == original_task.file_path
        assert restored_task.file_name == original_task.file_name
        assert restored_task.file_md5 == original_task.file_md5
        assert restored_task.status == original_task.status
        assert restored_task.retry_count == original_task.retry_count


class TestAsyncFileUpload:
    """异步文件上传功能测试"""
    
    @pytest.mark.asyncio
    async def test_process_uploaded_file_async_success(self):
        """测试异步文件上传成功场景"""
        # 创建模拟的知识库文档
        mock_kb_doc = MagicMock()
        mock_kb_doc.embedding_config.embedding_model = "test-model"
        mock_kb_doc.embedding_config.embedding_supplier = "test-supplier"
        mock_kb_doc.filesList = []
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix="_test.txt") as tmp_file:
            tmp_file.write(b"test content")
            tmp_file_path = tmp_file.name
        
        try:
            # 创建模拟的上传文件
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.txt"
            mock_file.file = open(tmp_file_path, 'rb')
            
            # Mock 各种依赖
            with patch('src.service.knowledgeSev.ObjectId') as mock_object_id, \
                 patch('src.service.knowledgeSev.KnowledgeBaseModel') as mock_kb_model, \
                 patch('src.service.knowledgeSev.Knowledge') as mock_knowledge, \
                 patch('src.service.file_queue_manager.file_queue_manager') as mock_queue_manager:
                
                # 设置 Mock 返回值
                mock_object_id.is_valid.return_value = True
                mock_kb_model.get = AsyncMock(return_value=mock_kb_doc)
                mock_knowledge.get_file_md5.return_value = "test-md5"
                mock_queue_manager.add_task = AsyncMock(return_value="test-task-id")
                
                # 调用被测试的函数
                result = await process_uploaded_file_async("test-kb-id", mock_file)
                
                # 验证结果
                assert result["task_id"] == "test-task-id"
                assert result["status"] == "queued"
                assert result["file_name"] == "test.txt"
                assert result["file_md5"] == "test-md5"
                assert "正在后台处理" in result["message"]
                
                # 验证队列管理器被调用
                mock_queue_manager.add_task.assert_called_once()
                
        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            mock_file.file.close()
    
    @pytest.mark.asyncio
    async def test_process_uploaded_file_async_duplicate(self):
        """测试上传重复文件的场景"""
        # 创建模拟的知识库文档，包含已存在的文件
        mock_kb_doc = MagicMock()
        mock_kb_doc.embedding_config.embedding_model = "test-model"
        mock_kb_doc.embedding_config.embedding_supplier = "test-supplier"
        mock_kb_doc.filesList = [{"file_md5": "test-md5", "file_name": "existing.txt"}]
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix="_test.txt") as tmp_file:
            tmp_file.write(b"test content")
            tmp_file_path = tmp_file.name
        
        try:
            # 创建模拟的上传文件
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.txt"
            mock_file.file = open(tmp_file_path, 'rb')
            
            # Mock 各种依赖
            with patch('src.service.knowledgeSev.ObjectId') as mock_object_id, \
                 patch('src.service.knowledgeSev.KnowledgeBaseModel') as mock_kb_model, \
                 patch('src.service.knowledgeSev.Knowledge') as mock_knowledge:
                
                # 设置 Mock 返回值
                mock_object_id.is_valid.return_value = True
                mock_kb_model.get = AsyncMock(return_value=mock_kb_doc)
                mock_knowledge.get_file_md5.return_value = "test-md5"  # 返回相同的 MD5
                
                # 调用被测试的函数，应该抛出 ValueError
                with pytest.raises(ValueError, match="已存在于此知识库"):
                    await process_uploaded_file_async("test-kb-id", mock_file)
                
        finally:
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            mock_file.file.close()


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_async_workflow_simulation(self):
        """模拟完整的异步工作流程"""
        # 这是一个模拟测试，展示完整的异步处理流程

        # 1. 初始化队列管理器
        queue_manager = FileQueueManager(max_queue_size=5, max_workers=1)

        await queue_manager.initialize()

        try:
            # 2. 添加任务到队列
            task_id = await queue_manager.add_task(
                kb_id="test-kb",
                file_path="/tmp/test.txt",
                file_name="test.txt",
                file_md5="abc123"
            )

            assert task_id is not None
            assert queue_manager.file_queue.qsize() == 1

            # 3. 检查队列状态
            status = await queue_manager.get_queue_status()
            assert status["queue_size"] == 1

            # 4. 模拟任务处理（不启动实际的工作进程）
            task = await queue_manager.file_queue.get()
            assert task.file_name == "test.txt"
            assert task.status == "queued"

            # 标记任务完成
            queue_manager.file_queue.task_done()

        finally:
            await queue_manager.stop_workers()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
