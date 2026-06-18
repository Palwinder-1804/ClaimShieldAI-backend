import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.dependencies import get_db, get_current_user
from app.db.models import Claim, User, ClaimFeedback, AuditLog
from app.schemas.claim import ClaimFeedbackCreate, ClaimFeedbackResponse

router = APIRouter(prefix="/claims", tags=["Decision Feedback"])
logger = logging.getLogger(__name__)

@router.post("/{claim_id}/feedback", response_model=ClaimFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def submit_claim_feedback(
    claim_id: uuid.UUID,
    feedback_in: ClaimFeedbackCreate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Submits user ratings (1-5) and validation feedback for a concluded claim decision.
    """
    # Verify claim exists and belongs to user
    claim_result = await db.execute(select(Claim).where(Claim.id == claim_id))
    claim = claim_result.scalars().first()
    
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim record not found."
        )
        
    if user.role != "admin" and claim.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to submit feedback for this claim."
        )

    # Feedback can only be submitted on finalized claims
    if claim.status != "done":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback can only be submitted for completed claims."
        )

    # Check if feedback already exists
    fb_check = await db.execute(select(ClaimFeedback).where(ClaimFeedback.claim_id == claim_id))
    existing_fb = fb_check.scalars().first()
    if existing_fb:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Feedback has already been submitted for this claim decision."
        )

    # Create new feedback
    new_feedback = ClaimFeedback(
        claim_id=claim_id,
        user_id=user.id,
        rating=feedback_in.rating,
        agreed_with_decision=feedback_in.agreed_with_decision,
        comment=feedback_in.comment
    )
    db.add(new_feedback)
    await db.flush()

    # Log to audits
    audit = AuditLog(
        user_id=user.id,
        claim_id=claim_id,
        action="feedback_submitted",
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={
            "rating": feedback_in.rating,
            "agreed_with_decision": feedback_in.agreed_with_decision
        }
    )
    db.add(audit)
    
    await db.commit()
    logger.info(f"Feedback successfully saved for Claim {claim_id}")
    
    # Send validation feedback email
    from app.services.email_service import email_service
    try:
        await email_service.send_feedback_email(
            claim_id=str(claim_id),
            rating=feedback_in.rating,
            agreed=feedback_in.agreed_with_decision,
            comment=feedback_in.comment,
            submitter_email=user.email
        )
    except Exception as email_err:
        logger.error(f"Error sending validation feedback email: {email_err}")
    
    return new_feedback
