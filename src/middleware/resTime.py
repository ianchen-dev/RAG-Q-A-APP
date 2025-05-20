# 响应时间中间件
import logging
import time

from fastapi import Request

logger = logging.getLogger(__name__)


async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request completed in {process_time:.4f} seconds")
    return response
