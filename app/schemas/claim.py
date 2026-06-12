import uuid
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class ClaimCreate(BaseModel):
    claim_type: str = Field(..., min_length=2, max_length=50, description="e.g. health, motor, life, property")
    documents: List[str] = Field(default=[], description="List of file paths uploaded to MinIO storage")

class ClaimResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    status: str
    claim_type: str
    documents: List[str]
    fraud_score: Optional[float] = None
    decision: Optional[str] = None
    policy_match: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    created_at: datetime

class ClaimFeedbackCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Score from 1 to 5 stars")
    agreed_with_decision: bool
    comment: Optional[str] = Field(None, max_length=1000)

class ClaimFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    claim_id: uuid.UUID
    user_id: uuid.UUID
    rating: int
    agreed_with_decision: bool
    comment: Optional[str] = None
    created_at: datetime

class PolicyDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    file_path: str
    chunk_count: int
    indexed_at: datetime

class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    claim_id: Optional[uuid.UUID] = None
    action: str
    detail: Dict[str, Any]
    ip_address: str
    timestamp: datetime
