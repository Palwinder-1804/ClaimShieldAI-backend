from celery import Celery
from kombu import Queue, Exchange
from app.core.config import settings

# Initialize Celery app
celery_app = Celery(
    "claimshield_workers",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Durability and delivery configurations
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Acknowledge task only after success/failure (rather than before run)
    task_acks_late=True,
    # Reject task if worker crashes during execution so it gets re-queued
    task_reject_on_worker_lost=True,
    # Set worker prefetch multiplier to 1 for fair task distribution
    worker_prefetch_multiplier=1,
)

# Dead letter exchange and routing configuration
# If task fails after max retries, it routes to the dead_letter queue
default_exchange = Exchange("claims_exchange", type="direct")
dead_letter_exchange = Exchange("dead_letter_exchange", type="direct")

celery_app.conf.task_queues = (
    Queue("claims", default_exchange, routing_key="claims.process"),
    Queue("dead_letter", dead_letter_exchange, routing_key="claims.dead"),
)

celery_app.conf.task_default_queue = "claims"
celery_app.conf.task_default_exchange = "claims_exchange"
celery_app.conf.task_default_routing_key = "claims.process"
