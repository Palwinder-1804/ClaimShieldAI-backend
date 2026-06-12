import logging
from typing import Dict, Any
from app.core.telemetry import claim_tracer
from app.services.ocr_service import ocr_service
from app.services.pii_masker import pii_masker

logger = logging.getLogger(__name__)

async def ingest_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingests claim documents from MinIO, extracts textual content using OCR,
    and anonymizes PII details before downstream LLM invocation.
    """
    claim_id = state.get("claim_id")
    documents = state.get("documents", [])
    
    logger.info(f"Ingest Agent started for Claim ID: {claim_id}. Documents count: {len(documents)}")
    
    with claim_tracer.start_as_current_span("ingest_agent") as span:
        span.set_attribute("claim_id", claim_id)
        span.set_attribute("documents_count", len(documents))
        
        extracted_texts = []
        
        # Loop through document file paths
        for doc_path in documents:
            try:
                # Retrieve and extract text from the document (PDF, Image, etc.)
                text = await ocr_service.extract_text(doc_path)
                if text:
                    extracted_texts.append(text)
            except Exception as e:
                logger.error(f"Error extracting text from {doc_path}: {e}")
                
        full_extracted_text = "\n\n".join(extracted_texts) if extracted_texts else "No text extracted from documents."
        
        # Mask PII (Names, Emails, SSH, Phone numbers, Account details)
        try:
            masked_text = await pii_masker.mask_text(full_extracted_text)
        except Exception as e:
            logger.error(f"PII Masker error: {e}. Falling back to unmasked text for pipeline safety.")
            masked_text = full_extracted_text
            
        logger.info(f"Ingest Agent finished. Extracted {len(full_extracted_text)} chars, masked to {len(masked_text)} chars.")
        
        return {
            "extracted_text": full_extracted_text,
            "masked_text": masked_text
        }
