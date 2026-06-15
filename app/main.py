import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.telemetry import setup_telemetry
from app.api.auth import router as auth_router, limiter as auth_limiter
from app.api.claims import router as claims_router
from app.api.feedback import router as feedback_router
from app.api.policy import router as policy_router
from app.api.policy_chat import router as policy_chat_router

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Intelligent insurance claim fraud detection and RAG analysis system."
)

# Set up CORS middleware
origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # React + Vite default dev port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "https://claim-shield-ai-frontend.vercel.app"
]

if settings.FRONTEND_URL not in origins:
    origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Rate Limiter Exception Handler
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Handles API rate limit exhaustion and returns a 429 payload.
    """
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}. Please try again later."}
    )

# Set up OpenTelemetry Tracing
setup_telemetry(app)

# Set up Prometheus Metrics Instrumentor
try:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics endpoint successfully registered.")
except Exception as e:
    logger.error(f"Failed to register Prometheus instrumentator: {e}")

# Auto-scaffold database tables on startup (excellent for local quick-runs)
@app.on_event("startup")
async def startup_event():
    try:
        from app.db.models import Base
        from app.db.session import engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL database tables successfully verified/created.")
    except Exception as db_err:
        logger.error(f"Error during startup DB schema creation: {db_err}")

# Include Routers
app.include_router(auth_router)
app.include_router(claims_router)
app.include_router(feedback_router)
app.include_router(policy_router)
app.include_router(policy_chat_router)

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "docs_url": "/docs"
    }
