import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.dependencies import get_db, get_admin_user, get_current_user
from app.db.models import PolicyDocument, PolicyChatSession, User
from app.schemas.claim import PolicyDocumentResponse
from app.services.ocr_service import ocr_service
from app.rag.chunker import document_chunker
from app.rag.vector_store import vector_store_manager
from app.rag.retriever import policy_retriever
from app.services.llm_service import llm_service

router = APIRouter(prefix="/policy", tags=["Policy Management"])
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=PolicyDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_policy_document(
    file: UploadFile = File(...),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Admin endpoint to upload and index an official policy document (PDF).
    Splits, embeds, indexes via FAISS, and stores metadata in PostgreSQL.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF format documents are supported for policy extraction."
        )

    file_bytes = await file.read()
    
    # 1. Upload file to MinIO
    policy_id = str(uuid.uuid4())
    secure_filename = f"policies/{policy_id}.pdf"
    
    try:
        saved_path = await ocr_service.upload_document(
            object_name=secure_filename,
            file_data=file_bytes,
            content_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"Failed uploading policy document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to write policy file to object store."
        )

    # 2. Extract Text & Chunk
    try:
        text = await ocr_service.extract_text(saved_path)
        if not text or text.startswith("Scanned PDF Document:"):
            # Fallback direct bytes extract
            text = await ocr_service._extract_pdf_text(file_bytes)
            
        chunks = document_chunker.split_text(text)
        if not chunks:
            raise ValueError("No extractable chunks found inside document.")
    except Exception as e:
        logger.error(f"Policy parsing/chunking failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract document contents: {str(e)}"
        )

    # 3. Create FAISS Vector Index
    try:
        # Create metadata structure
        metadatas = [{"name": file.filename, "source": saved_path} for _ in chunks]
        await vector_store_manager.create_and_save_index(
            chunks=chunks,
            policy_id=policy_id,
            metadatas=metadatas
        )
    except Exception as e:
        logger.error(f"FAISS index builder failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed compiling FAISS indices: {str(e)}"
        )

    # 4. Save metadata to DB
    new_policy = PolicyDocument(
        id=uuid.UUID(policy_id),
        name=file.filename,
        file_path=saved_path,
        chunk_count=len(chunks)
    )
    db.add(new_policy)
    await db.commit()
    
    logger.info(f"Policy {file.filename} uploaded and indexed successfully into {policy_id}.")
    return new_policy


@router.get("", response_model=List[PolicyDocumentResponse])
async def list_policy_documents(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Lists all available indexed policy documents.
    """
    result = await db.execute(select(PolicyDocument).order_by(PolicyDocument.indexed_at.desc()))
    return result.scalars().all()


@router.get("/{policy_id}/claim-steps", response_model=dict)
async def get_policy_claiming_steps(
    policy_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the policy claiming steps (filing procedures) dynamically using RAG.
    """
    result = await db.execute(select(PolicyDocument).where(PolicyDocument.id == policy_id))
    policy = result.scalars().first()
    
    policy_name = ""
    if policy:
        policy_name = policy.name
    else:
        # Fallback to PolicyChatSession for user-uploaded chat policies
        chat_result = await db.execute(select(PolicyChatSession).where(PolicyChatSession.id == policy_id))
        chat_session = chat_result.scalars().first()
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Policy document or chat session not found."
            )
        policy_name = chat_session.uploaded_filename

    # Query hybrid RAG retriever for claim filing steps
    query = "claim submission procedure steps how to file a claim requirements deadlines"
    try:
        matched_clauses = await policy_retriever.retrieve_clauses(query, policy_id=str(policy_id))
    except Exception as e:
        logger.error(f"Failed to retrieve clauses for claiming steps: {e}")
        matched_clauses = []

    if not matched_clauses:
        return {
            "policy_id": policy_id,
            "policy_name": policy_name,
            "steps": "No explicit claiming steps found in this policy document."
        }

    # LLM compiles a clean step-by-step guide
    context = "\n\n".join([c["text"] for c in matched_clauses])
    system_prompt = (
        "You are a helpful insurance claims assistant.\n"
        "Based on the policy clauses context, compile a clear, numbered step-by-step guide "
        "on how to file a claim, including any requirements, forms, deadlines, or contact info.\n"
        "Return the response in raw Markdown formatting, without any code block wrappers."
    )
    
    try:
        claiming_steps = await llm_service.call_llm(
            prompt=f"{system_prompt}\n\nContext:\n{context}",
            max_tokens=800
        )
    except Exception as e:
        logger.error(f"Error compiling claiming steps: {e}")
        claiming_steps = "Error generating claiming steps dynamically."

    return {
        "policy_id": policy_id,
        "policy_name": policy_name,
        "steps": claiming_steps.strip()
    }
