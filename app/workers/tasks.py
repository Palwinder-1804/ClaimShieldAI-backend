import asyncio
import logging
import uuid
from sqlalchemy.future import select
from celery.exceptions import MaxRetriesExceededError

from app.workers.celery_app import celery_app
from app.db.session import AsyncSessionLocal
from app.db.models import Claim, AuditLog
from app.orchestrator.graph import ClaimPipeline
from app.core.config import settings
from app.services.email_service import email_service

logger = logging.getLogger(__name__)

def run_async(coro):
    """
    Helper to run async coroutines inside Celery's synchronous thread environment.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(bind=True, max_retries=3)
def process_claim_task(self, claim_uuid_str: str):
    """
    Celery task that invokes the LangGraph claims processing pipeline.
    Re-runs up to 3 times on transient errors, routing to DLQ and alerts on absolute failures.
    """
    logger.info(f"Celery worker picked up Claim: {claim_uuid_str} (Attempt {self.request.retries + 1})")
    
    try:
        claim_id = uuid.UUID(claim_uuid_str)
    except ValueError:
        logger.error(f"Malformed Claim UUID string passed to worker: {claim_uuid_str}")
        return

    # Run the processing pipeline
    try:
        run_async(process_claim_core(claim_id))
    except Exception as exc:
        logger.error(f"Error processing claim {claim_id}: {exc}")
        try:
            # Requeue with a 60s retry delay
            raise self.retry(exc=exc, countdown=60)
        except MaxRetriesExceededError:
            logger.critical(f"Claim {claim_id} failed permanently after 3 attempts. Routing to DLQ.")
            run_async(handle_permanent_failure(claim_id, str(exc)))


async def process_claim_core(claim_id: uuid.UUID) -> None:
    """
    Core pipeline logic updating PostgreSQL DB.
    """
    # 1. Fetch claim and set status to processing, then close session
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Claim).where(Claim.id == claim_id))
        claim = result.scalars().first()
        if not claim:
            logger.error(f"Claim record not found in DB: {claim_id}")
            return
            
        claim.status = "processing"
        documents = claim.documents
        user_id = claim.user_id
        claim_type = claim.claim_type
        await db.commit()

    # 2. Invoke LangGraph claims pipeline (slow external API / LLM / OCR task)
    # No database connection is held checked-out during this await!
    final_state = await ClaimPipeline.process_claim(str(claim_id), documents, claim_type)
    
    # 3. Open a new session to map outputs to DB and log audit
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Claim).where(Claim.id == claim_id))
        claim = result.scalars().first()
        if not claim:
            logger.error(f"Claim record not found in DB when trying to save results: {claim_id}")
            return
            
        claim.fraud_score = final_state.get("fraud_score", 0.0)
        claim.decision = final_state.get("decision", "investigate")
        claim.report = final_state.get("report", "")
        # Store policy clauses as policy_match JSON
        claim.policy_match = {
            "policy_id": final_state.get("policy_id", ""),
            "matched_clauses_count": len(final_state.get("policy_clauses", [])),
            "clauses": final_state.get("policy_clauses", []),
            "claim_analysis": final_state.get("claim_analysis", {})
        }
        claim.status = "done"
        
        # Log audit
        audit = AuditLog(
            user_id=user_id,
            claim_id=claim_id,
            action="claim_processed",
            ip_address="127.0.0.1",
            detail={"decision": claim.decision, "fraud_score": claim.fraud_score}
        )
        db.add(audit)
        await db.commit()
        logger.info(f"Database successfully updated for Claim {claim_id}")


async def handle_permanent_failure(claim_id: uuid.UUID, error_message: str) -> None:
    """
    Updates the claim status to 'failed' and notifies administrators & users via SMTP email.
    """
    async with AsyncSessionLocal() as db:
        from sqlalchemy.orm import selectinload
        result = await db.execute(
            select(Claim)
            .options(selectinload(Claim.user))
            .where(Claim.id == claim_id)
        )
        claim = result.scalars().first()
        if claim:
            claim.status = "failed"
            claim.report = f"# Claim Processing Failure\n\nSystem failed to process this claim. Details: {error_message}"
            
            audit = AuditLog(
                user_id=claim.user_id,
                claim_id=claim_id,
                action="claim_failed_permanently",
                ip_address="127.0.0.1",
                detail={"error": error_message}
            )
            db.add(audit)
            await db.commit()
            
            # Send email to the user who filed the claim
            if claim.user and claim.user.email:
                await email_service.send_claim_failure_email(
                    email_to=claim.user.email,
                    claim_id=str(claim_id),
                    error_message=error_message
                )
            
            # Send critical alert alert to system administrator
            await email_service.send_admin_failure_alert(
                claim_id=str(claim_id),
                error_message=error_message
            )


async def process_claim_locally(claim_id: uuid.UUID) -> None:
    """
    Development fallback pipeline runner that runs locally in-process without Celery.
    """
    logger.info(f"Running development sync-fallback processor for Claim {claim_id}")
    try:
        await process_claim_core(claim_id)
    except Exception as e:
        logger.error(f"Fallback local processing failed: {e}")
        await handle_permanent_failure(claim_id, str(e))
