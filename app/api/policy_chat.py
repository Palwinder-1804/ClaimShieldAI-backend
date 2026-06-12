import os
import uuid
import logging
import hashlib
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.dependencies import get_db, get_current_user
from app.db.models import PolicyChatSession, User, PolicyDocument, AuditLog
from app.services.ocr_service import ocr_service
from app.services.pii_masker import pii_masker
from app.services.llm_service import llm_service
from app.services.session_service import redis_client
from app.rag.chunker import document_chunker
from app.rag.vector_store import vector_store_manager
from app.rag.retriever import policy_retriever

router = APIRouter(prefix="/api/policy-chat", tags=["Policy Chat Assistant"])
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str
    policy_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    confidence: float

@router.post("/upload", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_and_index_policy_chat(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    User endpoint to upload a personal policy PDF for Q&A.
    Performs OCR, masks PII, chunks, FAISS indexes, and creates a chat session.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF format files can be uploaded for chat index analysis."
        )

    # Limit to 20MB
    MAX_FILE_SIZE = 20 * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the maximum upload limit of 20MB."
        )

    # 1. Upload to MinIO
    policy_id = str(uuid.uuid4())
    secure_filename = f"chat_policies/{policy_id}.pdf"
    
    try:
        saved_path = await ocr_service.upload_document(
            object_name=secure_filename,
            file_data=file_bytes,
            content_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"Failed uploading chat policy PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed writing policy document to backend store."
        )

    # 2. Extract Text -> PII Mask -> Chunk
    try:
        text = await ocr_service.extract_text(saved_path)
        # Check empty page cases
        if not text or text.startswith("Scanned PDF Document:"):
            text = await ocr_service._extract_pdf_text(file_bytes)
            
        masked_text = await pii_masker.mask_text(text)
        chunks = document_chunker.split_text(masked_text)
        if not chunks:
            raise ValueError("Document does not contain any extractable text content.")
    except Exception as e:
        logger.error(f"Failed to process chat policy document text: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed parsing document contents: {str(e)}"
        )

    # 3. Create FAISS vector index specifically for this policy_id session
    try:
        metadatas = [{"name": file.filename, "source": saved_path} for _ in chunks]
        await vector_store_manager.create_and_save_index(
            chunks=chunks,
            policy_id=policy_id,
            metadatas=metadatas
        )
    except Exception as e:
        logger.error(f"Failed constructing FAISS index for chat policy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed compiling search index: {str(e)}"
        )

    # 4. Save metadata session to DB
    chat_session = PolicyChatSession(
        id=uuid.UUID(policy_id),
        user_id=user.id,
        uploaded_filename=file.filename,
        chunk_count=len(chunks)
    )
    db.add(chat_session)
    
    # Audit log
    audit = AuditLog(
        user_id=user.id,
        action="chat_policy_uploaded",
        ip_address=request.client.host if request.client else "127.0.0.1",
        detail={"filename": file.filename, "policy_id": policy_id}
    )
    db.add(audit)
    
    await db.commit()
    logger.info(f"Custom chat policy {file.filename} uploaded and indexed as session {policy_id}.")
    
    return {
        "policy_id": policy_id,
        "uploaded_filename": file.filename,
        "chunk_count": len(chunks)
    }


@router.post("/ask", response_model=ChatResponse)
async def ask_policy_assistant(
    chat_req: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Core Q&A logic. Verifies if query is insurance-related.
    Applies RAG if policy_id provided, otherwise answers with general insurance concepts.
    Caches outputs in Redis (1hr TTL).
    """
    question = chat_req.question.strip()
    policy_id = chat_req.policy_id
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. Guardrail Classifier: Enforce insurance-related topics only
    classification_prompt = (
        "You are an insurance classifier system. Analyze the user's question.\n"
        "Determine if it is related to insurance concepts, policy coverages, deductibles, premiums, "
        "filing insurance claims, risk assessments, actuarial calculations, policy exclusions, or general insurance topics.\n"
        "Return ONLY 'YES' if it is insurance-related, or 'NO' if it is completely unrelated to insurance.\n"
        "Do not include any punctuation or extra words.\n\n"
        f"Question: {question}"
    )
    
    classification = await llm_service.call_llm(classification_prompt, max_tokens=10)
    classification_cleaned = classification.strip().upper()
    
    if "NO" in classification_cleaned and "YES" not in classification_cleaned:
        logger.info(f"Guardrail trigger: Blocked non-insurance question: '{question}'")
        return ChatResponse(
            answer="ClaimShield Chat can only answer insurance-related questions. Please ask about policy details, coverages, claims, premiums, deductibles, or general insurance topics.",
            sources=[],
            confidence=0.0
        )

    # 2. Check Redis Cache
    cache_scope = policy_id if policy_id else "general"
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    cache_key = f"chat_cache:{cache_scope}:{question_hash}"
    
    try:
        cached_response = await redis_client.get(cache_key)
        if cached_response:
            logger.info(f"Chat Cache HIT for key: {cache_key}")
            cached_data = json.loads(cached_response)
            return ChatResponse(**cached_data)
    except Exception as e:
        logger.warning(f"Redis chat cache fetch failed: {e}")

    # 3. Retrieve Answer (RAG vs General)
    sources = []
    confidence = 0.70  # Default general confidence
    
    if policy_id:
        # Document-grounded Q&A
        try:
            # Query custom policy FAISS
            matched_clauses = await policy_retriever.retrieve_clauses(question, policy_id=policy_id)
            if matched_clauses:
                # Compile retrieved context
                context = "\n\n".join([f"Clause (Source: {c['doc_name']}):\n{c['text']}" for c in matched_clauses])
                sources = [{"doc_name": c["doc_name"], "snippet": c["text"][:300] + "..."} for c in matched_clauses]
                
                system_prompt = (
                    "You are a helpful insurance policy chat assistant. Your job is to answer the user's question "
                    "using ONLY the provided policy clauses context.\n"
                    "Do not make up facts or use external knowledge. Be precise. If the answer cannot be found "
                    "in the context, state that clearly.\n\n"
                    f"Context:\n{context}"
                )
                
                answer = await llm_service.call_llm(
                    prompt=f"{system_prompt}\n\nQuestion: {question}",
                    max_tokens=1000
                )
                # Compute confidence heuristic based on hybrid retriever top score
                confidence = float(min(matched_clauses[0]["score"] + 0.1, 1.0))
            else:
                # Policy loaded, but retriever returned empty results
                answer = "No matching policy clauses could be found in the index for this query. However, generally speaking:"
                general_answer = await self._get_general_answer(question)
                answer += f"\n\n{general_answer}"
                confidence = 0.50
        except Exception as e:
            logger.error(f"RAG search error for Q&A session {policy_id}: {e}")
            answer = "An error occurred searching the document index. Here is a general answer to your query:"
            general_answer = await _get_general_answer(question)
            answer += f"\n\n{general_answer}"
            confidence = 0.40
    else:
        # General insurance query
        answer = await _get_general_answer(question)

    response_data = ChatResponse(
        answer=answer,
        sources=sources,
        confidence=round(confidence, 2)
    )

    # 4. Cache answer in Redis with 1 hour TTL
    try:
        await redis_client.setex(cache_key, 3600, json.dumps(response_data.model_dump()))
        logger.info(f"Saved response in Redis chat cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache chat response: {e}")

    return response_data


async def _get_general_answer(question: str) -> str:
    """
    Calls LLM with general insurance prompt.
    """
    system_prompt = (
        "You are an objective insurance expert. The user is asking a general insurance question.\n"
        "Answer the question using standard industry definitions (deductibles, coverage limits, claims, premiums, exclusions).\n"
        "Do not references specific proprietary policy details. Keep the tone helpful, clear, and objective."
    )
    return await llm_service.call_llm(f"{system_prompt}\n\nQuestion: {question}", max_tokens=1000)


@router.delete("/session/{policy_id}", response_model=dict)
async def clear_chat_session(
    policy_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Clears the custom FAISS index files from disk and removes the session from the DB.
    """
    session_result = await db.execute(
        select(PolicyChatSession).where(PolicyChatSession.id == uuid.UUID(policy_id))
    )
    session = session_result.scalars().first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
        
    if session.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied to close this session.")

    # Remove session index points from Qdrant
    try:
        await vector_store_manager.delete_index(policy_id)
    except Exception as e:
        logger.error(f"Failed deleting Qdrant index for session {policy_id}: {e}")

    # Remove from DB
    await db.delete(session)
    await db.commit()
    
    # We can clear the Redis keys (scan keys and delete)
    try:
        keys = await redis_client.keys(f"chat_cache:{policy_id}:*")
        if keys:
            await redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cached chat responses from Redis.")
    except Exception as e:
        logger.warning(f"Error flushing Redis keys for session: {e}")

    return {"detail": "Chat session successfully cleared and index files removed."}
