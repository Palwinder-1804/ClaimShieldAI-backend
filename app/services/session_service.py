import uuid
import logging
from typing import Optional
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize connection pool for Redis
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

class SessionService:
    @staticmethod
    async def create_session(user_id: str, ttl_seconds: int = 86400) -> Optional[str]:
        """
        Creates a session in Redis pointing session:{session_id} to the user_id.
        Defaults to 24 hours expiry.
        """
        session_id = str(uuid.uuid4())
        key = f"session:{session_id}"
        try:
            await redis_client.setex(key, ttl_seconds, user_id)
            return session_id
        except Exception as e:
            logger.error(f"Redis error creating session: {e}")
            return None

    @staticmethod
    async def get_user_by_session(session_id: str) -> Optional[str]:
        """
        Retrieves user_id associated with the session_id. Returns None if session expired or missing.
        """
        if not session_id:
            return None
        key = f"session:{session_id}"
        try:
            user_id = await redis_client.get(key)
            return user_id
        except Exception as e:
            logger.error(f"Redis error fetching session: {e}")
            return None

    @staticmethod
    async def delete_session(session_id: str) -> bool:
        """
        Deletes the session from Redis.
        """
        if not session_id:
            return False
        key = f"session:{session_id}"
        try:
            result = await redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis error deleting session: {e}")
            return False

session_service = SessionService()
