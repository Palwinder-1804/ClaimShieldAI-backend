import os
import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.dependencies import get_db, get_current_user
from app.db.models import Claim, User, AuditLog, ClaimFeedback
from app.schemas.claim import ClaimCreate, ClaimResponse
from app.services.ocr_service import ocr_service

# Import Celery delay trigger
try:
    from app.workers.tasks import process_claim_task
    CELERY_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Celery tasks modules not fully initialized. Local sync fallback pipeline will be used.")
    CELERY_AVAILABLE = False

router = APIRouter(prefix="/claims", tags=["Claims Processing"])
limiter = Limiter(key_func=get_remote_address)
logger = logging.getLogger(__name__)

def log_audit(db: AsyncSession, user_id: uuid.UUID, claim_id: uuid.UUID, action: str, ip_address: str, detail: dict):
    audit = AuditLog(
        user_id=user_id,
        claim_id=claim_id,
        action=action,
        ip_address=ip_address,
        detail=detail
    )
    db.add(audit)

@router.post("/upload-doc", response_model=dict)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts multipart document file upload, stores it securely in MinIO,
    and returns the saved file key/path.
    """
    # 20 MB file size limit
    MAX_FILE_SIZE = 20 * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the maximum limit of 20 Megabytes."
        )

    # Generate unique secure file key
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
    unique_key = f"user_{user.id}/{uuid.uuid4()}{file_ext}"
    
    # Upload via OCR Service
    try:
        saved_key = await ocr_service.upload_document(
            object_name=unique_key,
            file_data=file_bytes,
            content_type=file.content_type or "application/pdf"
        )
    except Exception as e:
        logger.error(f"File upload processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving document on storage volumes."
        )
        
    return {"file_path": saved_key, "filename": file.filename}


@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_claim(
    request: Request,
    claim_in: ClaimCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submits a claim with associated documents. Sets status to 'pending'
    and triggers the background LangGraph pipeline worker.
    """
    if not claim_in.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document must be uploaded to submit a claim."
        )

    # Create DB entry
    new_claim = Claim(
        user_id=user.id,
        status="pending",
        claim_type=claim_in.claim_type,
        documents=claim_in.documents
    )
    db.add(new_claim)
    await db.flush()

    log_audit(
        db=db,
        user_id=user.id,
        claim_id=new_claim.id,
        action="claim_submitted",
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={"claim_type": claim_in.claim_type, "documents_count": len(claim_in.documents)}
    )

    # Trigger async background tasks
    if CELERY_AVAILABLE:
        try:
            process_claim_task.delay(str(new_claim.id))
            logger.info(f"Enqueued claim processing task on Celery for Claim {new_claim.id}")
        except Exception as e:
            logger.error(f"Failed to submit Celery queue task: {e}. Running local async fallback.")
            new_claim.status = "processing"
            # Fallback to run locally on thread if Celery is completely offline in development
            import asyncio
            from app.workers.tasks import process_claim_locally
            asyncio.create_task(process_claim_locally(new_claim.id))
    else:
        # Fallback runner
        new_claim.status = "processing"
        import asyncio
        from app.workers.tasks import process_claim_locally
        asyncio.create_task(process_claim_locally(new_claim.id))

    await db.commit()
    return new_claim


@router.get("", response_model=List[ClaimResponse])
async def list_claims(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns list of submitted claims. Users view their own; admins view all.
    """
    if user.role == "admin":
        result = await db.execute(select(Claim).order_by(Claim.created_at.desc()))
    else:
        result = await db.execute(
            select(Claim).where(Claim.user_id == user.id).order_by(Claim.created_at.desc())
        )
    return result.scalars().all()


@router.get("/{claim_id}", response_model=dict)
async def get_claim(
    claim_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the detailed claim information including feedback if submitted.
    """
    result = await db.execute(select(Claim).where(Claim.id == claim_id))
    claim = result.scalars().first()
    
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim record not found."
        )

    # Access control
    if user.role != "admin" and claim.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied to access this claim record."
        )

    # Load feedback relation manually to avoid eager-loading errors
    fb_result = await db.execute(select(ClaimFeedback).where(ClaimFeedback.claim_id == claim_id))
    feedback = fb_result.scalars().first()
    
    claim_data = ClaimResponse.model_validate(claim).model_dump()
    
    if feedback:
        claim_data["feedback"] = {
            "rating": feedback.rating,
            "agreed_with_decision": feedback.agreed_with_decision,
            "comment": feedback.comment,
            "created_at": feedback.created_at
        }
    else:
        claim_data["feedback"] = None

    return claim_data


@router.get("/{claim_id}/status", response_model=dict)
async def get_claim_status(
    claim_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Simplified polling target for claim processing state.
    """
    result = await db.execute(select(Claim).where(Claim.id == claim_id))
    claim = result.scalars().first()
    
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim record not found."
        )
        
    if user.role != "admin" and claim.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied."
        )
        
    return {
        "claim_id": claim.id,
        "status": claim.status,
        "decision": claim.decision
    }


@router.delete("/{claim_id}", response_model=dict)
async def delete_claim(
    claim_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes a claim record from the database.
    """
    result = await db.execute(select(Claim).where(Claim.id == claim_id))
    claim = result.scalars().first()
    
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim record not found."
        )

    # Access control: users can only delete their own; admins can delete any
    if user.role != "admin" and claim.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied to delete this claim."
        )

    await db.delete(claim)
    await db.commit()
    return {"message": "Claim successfully deleted."}
