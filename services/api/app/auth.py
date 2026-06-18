"""
JWT authentication helpers.

Credentials are read from env vars:
  API_USERNAME   (default: admin)
  API_PASSWORD   (default: changeme  — override in production)
  JWT_SECRET_KEY (default: dev-secret — MUST be overridden in production)
  JWT_EXPIRE_HOURS (default: 24)
"""
import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

_SECRET_KEY   = os.getenv("JWT_SECRET_KEY",    "dev-secret-change-in-production")
_ALGORITHM    = "HS256"
_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
_USERNAME     = os.getenv("API_USERNAME", "admin")
_PASSWORD     = os.getenv("API_PASSWORD", "changeme")

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_oauth2  = OAuth2PasswordBearer(tokenUrl="/token")

# Pre-hash the password at startup so we never compare plaintext
_HASHED_PASSWORD = _pwd_ctx.hash(_PASSWORD)


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=_EXPIRE_HOURS)
    return jwt.encode({"sub": subject, "exp": expire}, _SECRET_KEY, algorithm=_ALGORITHM)


def authenticate_user(username: str, password: str) -> bool:
    return username == _USERNAME and _pwd_ctx.verify(password, _HASHED_PASSWORD)


def get_current_user(token: str = Depends(_oauth2)) -> str:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        subject: str | None = payload.get("sub")
        if subject is None:
            raise credentials_error
    except JWTError:
        raise credentials_error
    return subject
