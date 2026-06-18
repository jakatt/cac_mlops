"""POST /token — issue a JWT access token."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..auth import authenticate_user, create_access_token

router = APIRouter(tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str


@router.post("/token", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()) -> Token:
    """
    Exchange username + password for a JWT Bearer token.

    Use the returned `access_token` in the `Authorization: Bearer <token>` header
    to call protected endpoints (e.g. POST /predict).
    """
    if not authenticate_user(form.username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form.username)
    return Token(access_token=token, token_type="bearer")
