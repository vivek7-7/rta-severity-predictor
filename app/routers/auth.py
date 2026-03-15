"""
app/routers/auth.py
Authentication routes: register, login, logout.
JWT stored in httpOnly cookie.  All pages except /login redirect here if no cookie.
"""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.schemas.user import TokenData

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="app/templates")

# ── Password hashing ───────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT helpers ────────────────────────────────────────────────────────────────
def create_access_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> TokenData | None:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return TokenData(email=email)
    except JWTError:
        return None


# ── FastAPI dependency: require authenticated user ─────────────────────────────
async def get_current_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> User:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": f"/login?next={request.url.path}"},
        )
    token_data = decode_token(token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    result = await db.execute(select(User).where(User.email == token_data.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    return user


async def get_current_user_or_none(
    request: Request, db: AsyncSession = Depends(get_db)
) -> User | None:
    """Soft version — returns None instead of redirecting."""
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.get(
    "/login",
    response_class=HTMLResponse,
    summary="Login / Register page",
    description="Renders the sign-in and create-account form (Alpine.js tabs).",
)
async def login_page(request: Request, next: str = "/dashboard", error: str = ""):
    return templates.TemplateResponse(
        "login.html", {"request": request, "next": next, "error": error}
    )


@router.post(
    "/login",
    summary="Process login form",
    description="Validates credentials and sets httpOnly JWT cookie.",
)
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form(default="/dashboard"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "next": next,
                "error": "Invalid email or password.",
                "tab": "login",
            },
            status_code=400,
        )

    token = create_access_token(user.email)
    response = RedirectResponse(url=next, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    logger.info("User logged in: %s", user.email)
    return response


@router.post(
    "/register",
    summary="Process registration form",
    description="Creates a new user account with bcrypt-hashed password.",
)
async def register_submit(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    # Validation
    errors = []
    if len(full_name.strip()) < 2:
        errors.append("Full name must be at least 2 characters.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm_password:
        errors.append("Passwords do not match.")

    if errors:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "next": "/dashboard", "error": " ".join(errors), "tab": "register"},
            status_code=400,
        )

    # Duplicate email check
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "next": "/dashboard", "error": "Email already registered.", "tab": "register"},
            status_code=400,
        )

    user = User(
        full_name=full_name.strip(),
        email=email.lower().strip(),
        hashed_password=hash_password(password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.email)
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    logger.info("New user registered: %s", user.email)
    return response


@router.get(
    "/logout",
    summary="Logout",
    description="Clears the JWT cookie and redirects to /login.",
)
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("access_token")
    return response
