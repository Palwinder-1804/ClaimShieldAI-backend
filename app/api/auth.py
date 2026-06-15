import urllib.parse
from typing import Optional
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.dependencies import get_db, get_current_user
from app.core.security import get_password_hash, verify_password, create_access_token, decode_access_token
from app.core.oauth import exchange_code_for_tokens, get_google_user_info
from app.db.models import User, AuditLog
from app.schemas.auth import UserRegister, UserLogin, UserOnboard, UserResponse, TokenResponse, MessageResponse
from app.services.session_service import session_service
from app.services.email_service import email_service

router = APIRouter(prefix="/auth", tags=["Authentication"])
limiter = Limiter(key_func=get_remote_address)

def log_audit_action(db: AsyncSession, user_id: Optional[str], action: str, ip_address: str, detail: dict):
    """
    Helper to record events in the audit log.
    """
    # Since db session commits are handled in dependencies/routes, we can add to session directly
    audit = AuditLog(
        user_id=user_id,
        action=action,
        ip_address=ip_address,
        detail=detail
    )
    db.add(audit)

@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/minute")
async def register(request: Request, user_in: UserRegister, db: AsyncSession = Depends(get_db)):
    """
    Registers a new user, hashes the password, and sends a verification email.
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists in the system."
        )

    # Create new user
    hashed_pwd = get_password_hash(user_in.password)
    new_user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=hashed_pwd,
        auth_provider="email",
        is_verified=False,
        is_onboarded=False,
        role="user"
    )
    db.add(new_user)
    await db.flush()  # Populates id prior to committing

    # Create email verification token
    verification_token = create_access_token(
        data={"sub": str(new_user.id), "purpose": "email_verification"},
        expires_delta=timedelta(hours=24)
    )

    # Log action
    log_audit_action(
        db=db,
        user_id=new_user.id,
        action="user_registered",
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={"email": user_in.email}
    )

    # Send email
    await email_service.send_verification_email(new_user.email, verification_token)

    return {"detail": "Registration successful. Please check your email for the verification link."}


@router.get("/verify", response_model=MessageResponse)
async def verify_email(token: str, db: AsyncSession = Depends(get_db)):
    """
    Verifies the user's email using the registration token.
    """
    payload = decode_access_token(token)
    if not payload or payload.get("purpose") != "email_verification":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token."
        )
    
    user_id_str = payload.get("sub")
    import uuid
    try:
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed verification token payload."
        )

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    if user.is_verified:
        return {"detail": "Email already verified."}

    user.is_verified = True
    await db.commit()

    return {"detail": "Email verified successfully. You can now log in."}


@router.post("/login", response_model=UserResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Validates user credentials, creates a JWT access token, spins up a Redis session,
    and returns httpOnly cookies.
    """
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalars().first()
    
    if not user or user.auth_provider != "email":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    if not user.hashed_password or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please verify your email address before logging in."
        )

    # Generate session & token
    # Calculate cookie/session max ages (Remember Me -> 30 days)
    access_token_max_age = 2592000 if credentials.remember_me else 3600
    session_id_max_age = 2592000 if credentials.remember_me else 86400

    session_id = await session_service.create_session(str(user.id), ttl_seconds=session_id_max_age)
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not establish session storage. Please try again."
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(seconds=access_token_max_age)
    )

    # Set cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=access_token_max_age,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE
    )
    
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=session_id_max_age,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE
    )

    log_audit_action(
        db=db,
        user_id=user.id,
        action="user_logged_in",
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={"email": user.email}
    )

    return user


@router.get("/google")
async def google_login():
    """
    Redirects the user to Google OAuth2 Consent Screen.
    """
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    google_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=google_url)


@router.get("/google/callback")
async def google_callback(request: Request, response: Response, code: str, db: AsyncSession = Depends(get_db)):
    """
    Exchanges authorization code, registers/logs in the user, sets cookies,
    and redirects the browser to the frontend dashboard or onboarding.
    """
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth is not configured on this server."
        )

    token_data = await exchange_code_for_tokens(code)
    if not token_data or "access_token" not in token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to retrieve tokens from Google."
        )
        
    user_info = await get_google_user_info(token_data["access_token"])
    if not user_info or "email" not in user_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch profile details from Google."
        )

    email = user_info["email"]
    google_id = user_info.get("sub")
    full_name = user_info.get("name", email.split("@")[0])

    # Check if user already exists
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    
    if not user:
        # Create Google OAuth registered user
        user = User(
            email=email,
            full_name=full_name,
            auth_provider="google",
            google_id=google_id,
            is_verified=True,  # Google verified emails are pre-trusted
            is_onboarded=False,
            role="user"
        )
        db.add(user)
        await db.flush()
        action = "google_user_registered"
    else:
        # Link Google ID if email matches but Google ID not set
        if not user.google_id:
            user.google_id = google_id
            user.auth_provider = "google"
        action = "google_user_logged_in"

    # Establish session
    # For Google OAuth logins, we implicitly remember the user with a 30-day session
    google_max_age = 2592000
    session_id = await session_service.create_session(str(user.id), ttl_seconds=google_max_age)
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not establish session storage."
        )

    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(seconds=google_max_age)
    )

    # Redirect target (dynamically resolved using settings.FRONTEND_URL)
    frontend_redirect = f"{settings.FRONTEND_URL}/dashboard" if user.is_onboarded else f"{settings.FRONTEND_URL}/onboarding"
    redirect_response = RedirectResponse(url=frontend_redirect)

    # Set cookies on redirect response
    redirect_response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=google_max_age,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE
    )
    
    redirect_response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=google_max_age,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE
    )

    log_audit_action(
        db=db,
        user_id=user.id,
        action=action,
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={"email": email}
    )
    
    await db.commit()
    return redirect_response


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, response: Response):
    """
    Logs out the current user, invalidating the session in Redis and clearing client cookies.
    """
    session_id = request.cookies.get("session_id")
    if session_id:
        await session_service.delete_session(session_id)
        
    response.delete_cookie(key="access_token")
    response.delete_cookie(key="session_id")
    return {"detail": "Successfully logged out."}


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """
    Returns the currently logged in user profile.
    """
    return user


@router.patch("/me/onboard", response_model=UserResponse)
async def complete_onboarding(
    profile: UserOnboard,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Completes user profile onboarding and marks is_onboarded=True.
    """
    if user.is_onboarded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User onboarding has already been completed."
        )

    if profile.full_name:
        user.full_name = profile.full_name

    user.is_onboarded = True
    await db.commit()
    return user
