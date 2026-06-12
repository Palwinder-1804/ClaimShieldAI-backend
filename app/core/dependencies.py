from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.db.models import User
from app.core.security import decode_access_token
from app.services.session_service import session_service

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async database session, committing/rolling back transactions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Extracts access_token and session_id from request cookies.
    Decodes the JWT to verify signature and looks up the active session in Redis.
    Queries the user from the Supabase PostgreSQL database.
    """
    access_token = request.cookies.get("access_token")
    session_id = request.cookies.get("session_id")
    
    # Check headers fallback if cookies are not set (useful for API testing/Swagger docs)
    if not access_token and "Authorization" in request.headers:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            access_token = auth_header.replace("Bearer ", "", 1)
            
    if not session_id and "X-Session-ID" in request.headers:
        session_id = request.headers.get("X-Session-ID")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not access_token or not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials are missing (cookies or headers)",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 1. Decode access token JWT
    payload = decode_access_token(access_token)
    if payload is None:
        raise credentials_exception
        
    user_id_str: str = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception
        
    # 2. Check session existence in Redis
    redis_user_id = await session_service.get_user_by_session(session_id)
    if not redis_user_id or redis_user_id != user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please login again.",
        )
        
    # 3. Retrieve user from Database
    try:
        import uuid
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception
        
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalars().first()
    
    if user is None:
        raise credentials_exception
        
    return user

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Enforces role restriction to admin users.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation is restricted to administrator accounts.",
        )
    return current_user
