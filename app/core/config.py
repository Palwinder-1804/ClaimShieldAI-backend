import os
from typing import Optional
from pydantic import AnyHttpUrl, EmailStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core FastAPI Settings
    SECRET_KEY: str = "super-secret-jwt-key-change-in-production-1234567890"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    PROJECT_NAME: str = "ClaimShield AI"

    # Database Configuration (Supabase PostgreSQL)
    SUPABASE_URL: str = "https://your-project.supabase.co"
    SUPABASE_KEY: str = "your-supabase-anon-key"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:your-supabase-db-password@db.your-project.supabase.co:5432/postgres"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        # Enforce postgresql+asyncpg for async SQLAlchemy usage
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # Redis & Session Cache
    REDIS_URL: str = "redis://localhost:6379"

    # Google OAuth2 Credentials
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    BACKEND_URL: str = "http://localhost:8000"
    GOOGLE_REDIRECT_URI: Optional[str] = None

    # Frontend URL & Cookie Settings
    FRONTEND_URL: str = "http://localhost:5173"
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    @model_validator(mode="after")
    def set_google_redirect_uri(self) -> 'Settings':
        self.BACKEND_URL = self.BACKEND_URL.rstrip('/')
        if not self.GOOGLE_REDIRECT_URI:
            self.GOOGLE_REDIRECT_URI = f"{self.BACKEND_URL}/auth/google/callback"
        else:
            self.GOOGLE_REDIRECT_URI = self.GOOGLE_REDIRECT_URI.rstrip('/')
        return self

    # Email Configuration (fastapi-mail SMTP)
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: EmailStr = "noreply@claimshield.ai"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.example.com"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # MinIO Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "claims-documents"
    MINIO_ENCRYPT: bool = True
    MINIO_SECURE: bool = True

    # AI / ML Configuration
    OPENAI_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_LOCATION: Optional[str] = None
    QDRANT_COLLECTION: str = "claimshield_policies"

    # LangSmith Observability & Tracing
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "claimshield-ai"

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"

settings = Settings()

