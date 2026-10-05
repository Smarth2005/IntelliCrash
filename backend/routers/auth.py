"""
IntelliCrash — Authentication & Authorization Router.

Endpoints:
    POST /api/v1/auth/register   → Register a new user
    POST /api/v1/auth/login      → Login and receive JWT access token
    GET  /api/v1/auth/me         → Get current authenticated user profile
    GET  /api/v1/auth/users      → List all users (Admin only)
"""

from typing import List, Union
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.backend.config import get_settings
from src.backend.database import get_db
from src.backend.models import UserRegister, UserLogin, TokenResponse, UserResponse
from src.backend.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_roles,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Security"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, summary="Register new user")
def register_user(user_data: UserRegister):
    """
    Registers a new user account with hashed password and assigned role.
    Allowed roles: admin, operator, viewer.
    """
    if user_data.role not in ("admin", "operator", "viewer"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be one of: 'admin', 'operator', 'viewer'",
        )

    db = get_db()
    hashed = hash_password(user_data.password)

    with db.get_connection() as conn:
        # Check if username or email already exists
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (user_data.username, user_data.email),
        ).fetchone()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username or email already registered",
            )

        cursor = conn.execute(
            """
            INSERT INTO users (username, email, hashed_password, role, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (user_data.username, user_data.email, hashed, user_data.role),
        )
        conn.commit()
        user_id = cursor.lastrowid

        user_row = conn.execute(
            "SELECT id, username, email, role, is_active, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return dict(user_row)


@router.post("/login", response_model=TokenResponse, summary="Authenticate and acquire JWT access token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 standard login endpoint.
    Accepts form-data (username & password) and returns a signed JWT Bearer token.
    Compatible with Swagger UI 'Authorize' button.
    """
    db = get_db()
    with db.get_connection() as conn:
        user = conn.execute(
            "SELECT id, username, email, hashed_password, role, is_active FROM users WHERE username = ?",
            (form_data.username,),
        ).fetchone()

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    settings = get_settings()
    token_claims = {
        "sub": user["username"],
        "user_id": user["id"],
        "role": user["role"],
    }
    token = create_access_token(token_claims)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }


@router.get("/me", response_model=UserResponse, summary="Get current logged in user")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Returns the profile of the user authenticated by the Bearer token."""
    return current_user


@router.get("/users", response_model=List[UserResponse], summary="List all registered users (Admin only)")
async def list_users(admin_user: dict = Depends(require_roles("admin"))):
    """Admin-only endpoint to inspect all user accounts."""
    db = get_db()
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, email, role, is_active, created_at FROM users ORDER BY id"
        ).fetchall()

    return [dict(r) for r in rows]
