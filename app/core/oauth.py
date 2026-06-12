import httpx
from typing import Dict, Any, Optional
from app.core.config import settings

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

async def exchange_code_for_tokens(code: str) -> Optional[Dict[str, Any]]:
    """
    Exchanges the authorization code from Google OAuth callback for access and ID tokens.
    """
    data = {
        "code": code,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GOOGLE_TOKEN_URL, data=data, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError:
            return None

async def get_google_user_info(access_token: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the authenticated user's profile info from Google.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(GOOGLE_USERINFO_URL, headers=headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPError:
            return None
