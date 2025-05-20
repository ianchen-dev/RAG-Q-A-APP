from datetime import timedelta
from typing import Optional

from beanie import PydanticObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from src.models.user import User
from src.utils.jwt import ACCESS_TOKEN_EXPIRE_DAYS, create_access_token, decode_token
from src.utils.pwdHash import get_password_hash, verify_password


class UserIn(BaseModel):
    username: str = Field(max_length=50, description="用户名")
    password: str = Field(max_length=20, description="密码")
    email: str = Field(max_length=100, description="邮箱")


class UserLogin(BaseModel):
    username: str = Field(max_length=50, description="用户名")
    password: str = Field(max_length=20, description="密码")


class UserOut(BaseModel):
    id: str  # 修改为str类型，MongoDB使用ObjectId
    username: str
    email: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def user_login(userLogin: UserLogin):
    user = await User.find_one(User.username == userLogin.username)
    if not user or not verify_password(userLogin.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    user_data_for_token = {
        "sub": user.username,
        "email": user.email,
        "userId": str(user.id),
    }
    access_token = create_access_token(
        data=user_data_for_token, expires_delta=access_token_expires
    )
    user_out = UserOut(id=str(user.id), username=user.username, email=user.email)
    return {
        "token": access_token,
        "token_type": "bearer",
        "user": user_out.model_dump(),
    }


# Oauth2 验证
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    userId_str: Optional[str] = payload.get("userId")
    print(f"userID String: {userId_str}")
    if userId_str is None:
        raise credentials_exception

    try:
        user_object_id = PydanticObjectId(userId_str)
    except Exception:
        print(f"Error converting userId string '{userId_str}' to PydanticObjectId")
        raise credentials_exception

    user = await User.find_one(User.id == user_object_id)
    if user is None:
        print(f"User not found with ID: {user_object_id}")
        raise credentials_exception
    return user


async def user_register(userin: UserIn):
    hashed_password = get_password_hash(userin.password)
    user = User(
        username=userin.username,
        password=hashed_password,
        email=userin.email,
    )
    await user.insert()
    user_out = UserOut(id=str(user.id), username=user.username, email=user.email)
    return user_out.model_dump()
