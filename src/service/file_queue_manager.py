"""
文件处理异步队列管理器
实现文件上传的异步处理，提升响应速度和并发处理能力

主要功能：
1. 异步队列管理 - 使用 asyncio.Queue 实现文件处理任务队列
2. 多消费者并发处理 - 支持多个工作进程同时处理文件
3. 任务状态跟踪 - 内存存储任务状态，支持前端查询
4. 错误处理和重试 - 处理失败的任务支持重试机制
5. 优雅关闭 - 支持服务关闭时的队列清理

Author: hbchen
Date: 2025-09-19
"""

import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId

from src.models.knowledgeBase import KnowledgeBase as KnowledgeBaseModel
from src.service.knowledgeSev import process_uploaded_file_sync

# 配置日志
logger = logging.getLogger(__name__)


class FileProcessingTask:
    """文件处理任务模型"""

    def __init__(
        self,
        task_id: str,
        kb_id: str,
        file_path: str,
        file_name: str,
        file_md5: str,
        status: str = "queued",
        created_at: Optional[datetime] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        error_message: Optional[str] = None,
    ):
        self.task_id = task_id
        self.kb_id = kb_id
        self.file_path = file_path
        self.file_name = file_name
        self.file_md5 = file_md5
        self.status = status  # queued, processing, completed, failed
        self.created_at = created_at or datetime.now()
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.error_message = error_message

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "kb_id": self.kb_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_md5": self.file_md5,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FileProcessingTask":
        """从字典创建任务实例"""
        created_at = (
            datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None
        )
        return cls(
            task_id=data["task_id"],
            kb_id=data["kb_id"],
            file_path=data["file_path"],
            file_name=data["file_name"],
            file_md5=data["file_md5"],
            status=data.get("status", "queued"),
            created_at=created_at,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            error_message=data.get("error_message"),
        )


class FileQueueManager:
    """文件处理队列管理器"""

    def __init__(self, max_queue_size: int = 100, max_workers: int = 20):
        """
        初始化队列管理器

        Args:
            max_queue_size: 队列最大容量，防止内存溢出
            max_workers: 最大工作进程数量
        """
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.file_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self._tasks: Dict[str, FileProcessingTask] = {}

    async def initialize(self):
        """初始化队列管理器"""
        logger.info("文件队列管理器初始化成功")

    async def start_workers(self):
        """启动工作进程"""
        if self.is_running:
            logger.warning("工作进程已在运行中")
            return

        self.is_running = True
        logger.info(f"启动 {self.max_workers} 个文件处理工作进程")

        # 创建工作进程
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker_process(f"worker-{i}"))
            self.workers.append(worker_task)

        logger.info("所有文件处理工作进程已启动")

    async def stop_workers(self):
        """停止工作进程"""
        if not self.is_running:
            return

        logger.info("正在停止文件处理工作进程...")
        self.is_running = False

        # 取消所有工作进程
        for worker in self.workers:
            worker.cancel()

        # 等待所有工作进程完成
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("所有文件处理工作进程已停止")

    async def add_task(
        self, kb_id: str, file_path: str, file_name: str, file_md5: str
    ) -> str:
        """
        添加文件处理任务到队列

        Args:
            kb_id: 知识库ID
            file_path: 文件路径
            file_name: 文件名
            file_md5: 文件MD5值

        Returns:
            task_id: 任务ID

        Raises:
            asyncio.QueueFull: 队列已满
        """
        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务对象
        task = FileProcessingTask(
            task_id=task_id,
            kb_id=kb_id,
            file_path=file_path,
            file_name=file_name,
            file_md5=file_md5,
        )

        try:
            # 将任务添加到队列（非阻塞）
            self.file_queue.put_nowait(task)

            # 保存任务状态到内存
            await self._save_task_status(task)

            logger.info(f"任务 {task_id} 已添加到队列，文件: {file_name}")
            return task_id

        except asyncio.QueueFull:
            logger.error(f"队列已满，无法添加任务: {file_name}")
            raise
        except Exception as e:
            logger.error(f"添加任务失败: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务状态字典，如果任务不存在则返回 None
        """
        task = self._tasks.get(task_id)
        if task:
            return task.to_dict()
        return None

    async def get_queue_status(self) -> Dict:
        """
        获取队列状态信息

        Returns:
            队列状态字典
        """
        return {
            "queue_size": self.file_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "workers_count": len(self.workers),
            "is_running": self.is_running,
        }

    async def _worker_process(self, worker_name: str):
        """
        工作进程主循环

        Args:
            worker_name: 工作进程名称
        """
        logger.info(f"工作进程 {worker_name} 已启动")

        while self.is_running:
            try:
                # 从队列获取任务（阻塞等待）
                task = await asyncio.wait_for(
                    self.file_queue.get(),
                    timeout=1.0,  # 1秒超时，用于检查 is_running 状态
                )

                logger.info(f"工作进程 {worker_name} 开始处理任务 {task.task_id}")

                # 更新任务状态为处理中
                task.status = "processing"
                await self._save_task_status(task)

                # 处理文件
                success = await self._process_file_task(task, worker_name)

                if success:
                    task.status = "completed"
                    logger.info(f"工作进程 {worker_name} 完成任务 {task.task_id}")
                else:
                    # 处理失败，检查是否需要重试
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = "queued"
                        # 重新入队
                        await self.file_queue.put(task)
                        logger.warning(
                            f"任务 {task.task_id} 处理失败，重试第 {task.retry_count} 次"
                        )
                    else:
                        task.status = "failed"
                        logger.error(f"任务 {task.task_id} 处理失败，已达最大重试次数")

                # 保存最终状态
                await self._save_task_status(task)

                # 标记任务完成
                self.file_queue.task_done()

            except asyncio.TimeoutError:
                # 超时是正常的，用于检查 is_running 状态
                continue
            except asyncio.CancelledError:
                logger.info(f"工作进程 {worker_name} 被取消")
                break
            except Exception as e:
                logger.error(f"工作进程 {worker_name} 发生错误: {e}")
                logger.error(traceback.format_exc())
                # 继续运行，不退出工作进程

        logger.info(f"工作进程 {worker_name} 已退出")

    async def _process_file_task(
        self, task: FileProcessingTask, worker_name: str
    ) -> bool:
        """
        处理单个文件任务

        Args:
            task: 文件处理任务
            worker_name: 工作进程名称

        Returns:
            是否处理成功
        """
        try:
            # 验证知识库是否存在
            if not ObjectId.is_valid(task.kb_id):
                task.error_message = f"无效的知识库 ID 格式: {task.kb_id}"
                return False

            knowledge_base_doc = await KnowledgeBaseModel.get(ObjectId(task.kb_id))
            if not knowledge_base_doc:
                task.error_message = f"知识库 ID 未找到: {task.kb_id}"
                return False

            # 调用同步版本的文件处理函数
            await process_uploaded_file_sync(
                kb_id=task.kb_id,
                file_path=task.file_path,
                file_name=task.file_name,
                file_md5=task.file_md5,
                knowledge_base_doc=knowledge_base_doc,
            )

            logger.info(f"工作进程 {worker_name} 成功处理文件: {task.file_name}")
            return True

        except Exception as e:
            error_msg = f"处理文件 {task.file_name} 时发生错误: {str(e)}"
            task.error_message = error_msg
            logger.error(f"工作进程 {worker_name} - {error_msg}")
            logger.error(traceback.format_exc())
            return False

    async def _save_task_status(self, task: FileProcessingTask):
        """
        保存任务状态到内存

        Args:
            task: 文件处理任务
        """
        try:
            self._tasks[task.task_id] = task
        except Exception as e:
            logger.error(f"保存任务状态失败: {e}")


# 全局队列管理器实例
file_queue_manager = FileQueueManager()
