import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Boolean, Float, Integer, Text, ForeignKey, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Nullable for Google users
    auth_provider: Mapped[str] = mapped_column(String(50), default="email")  # "email" or "google"
    google_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_onboarded: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(50), default="user")  # "user" or "admin"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="user", cascade="all, delete-orphan")
    feedbacks: Mapped[List["ClaimFeedback"]] = relationship("ClaimFeedback", back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[List["PolicyChatSession"]] = relationship("PolicyChatSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="user")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending/processing/done/failed
    claim_type: Mapped[str] = mapped_column(String(50), nullable=False)  # health/motor/life/property
    documents: Mapped[dict] = mapped_column(JSON, default=list)  # list of file paths in MinIO
    fraud_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.0 - 1.0
    decision: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # approved/rejected/investigate
    policy_match: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # matched clauses
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # generated report markdown
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="claims")
    feedbacks: Mapped[List["ClaimFeedback"]] = relationship("ClaimFeedback", back_populates="claim", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="claim")


class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    chat_sessions: Mapped[List["PolicyChatSession"]] = relationship("PolicyChatSession", back_populates="policy_document")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    claim_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # "claim_submitted", "decision_made", etc.
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
    claim: Mapped[Optional["Claim"]] = relationship("Claim", back_populates="audit_logs")


class ClaimFeedback(Base):
    __tablename__ = "claim_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5
    agreed_with_decision: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="feedbacks")
    user: Mapped["User"] = relationship("User", back_populates="feedbacks")


class PolicyChatSession(Base):
    __tablename__ = "policy_chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    policy_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("policy_documents.id", ondelete="SET NULL"), nullable=True)
    uploaded_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="chat_sessions")
    policy_document: Mapped[Optional["PolicyDocument"]] = relationship("PolicyDocument", back_populates="chat_sessions")
