from typing import Optional

from beanie import Document, Indexed
from pydantic import Field


class User(Document):
    username: Indexed(str, unique=True) = Field(..., max_length=50)  # type:ignore
    password: str = Field(max_length=100)
    email: Indexed(str, unique=True) = Field(..., max_length=100)  # type:ignore
    nickname: Optional[str] = Field(None, max_length=50)

    class Settings:
        name = "users"

    def __str__(self):
        return self.username
