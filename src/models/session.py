from datetime import datetime
from typing import Annotated

from beanie import Document, Indexed


class Session(Document):
    title: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 会话标题
    username: Annotated[str, Indexed]  # 用户名
    assistant_id: str  # 助手ID
    created_at: datetime = datetime.now()  # 创建时间
    updated_at: datetime = datetime.now()  # 更新时间

    class Settings:
        name = "sessions"
        indexes = [
            [("updated_at", -1)],  # -1 表示降序索引
        ]
