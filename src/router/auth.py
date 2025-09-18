# src/routers/auth.py
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm  # 用于接收表单数据

from src.models.user import User  # 引入 User 模型
from src.utils.jwt import (  # 引入 Token 工具
    ACCESS_TOKEN_EXPIRE_DAYS,
    create_access_token,
)
from src.utils.pwdHash import verify_password  # 引入密码验证工具

AuthRouter = APIRouter()


@AuthRouter.post("/token")
async def login_for_access_token(
    form_data: Annotated[
        OAuth2PasswordRequestForm, Depends()
    ],  # 使用 Depends 获取表单数据
):
    """
    处理 OAuth2 Password Flow 的 Token 请求。
    接收 username 和 password 表单数据，验证用户并返回 access token。
    """
    # 1. 根据用户名查找用户
    user = await User.find_one(User.username == form_data.username)

    # 2. 验证用户是否存在以及密码是否正确
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            # 这个 headers 对于 OAuth2 Password Flow 很重要
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. (可选) 检查用户是否被禁用 (如果你的 User 模型有 is_active 或 disabled 字段)
    # if user.disabled:
    #     raise HTTPException(status_code=400, detail="Inactive user")

    # 4. 创建 Access Token
    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    # 准备要编码到 Token 中的数据
    user_data_for_token = {
        "sub": user.username,  # 'sub' (subject) 是 JWT 的标准字段，通常是用户名或用户ID
        "email": user.email,
        # 确保将 ObjectId 转换为字符串以便序列化到 JWT 中
        "userId": str(user.id),
    }
    access_token = create_access_token(
        data=user_data_for_token, expires_delta=access_token_expires
    )

    # 5. 返回 Token
    # 注意：标准的 OAuth2 响应包含 access_token 和 token_type
    return {"access_token": access_token, "token_type": "bearer"}
