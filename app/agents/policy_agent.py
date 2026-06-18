import logging
import json
from typing import Dict, Any
from app.core.telemetry import claim_tracer
from app.services.llm_service import llm_service
from app.rag.retriever import policy_retriever
import uuid
from sqlalchemy import select
from app.db.models import Claim

logger = logging.getLogger(__name__)

async def policy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the claim's masked text, extracts keywords for RAG,
    and searches the policy documents index for relevant coverage clauses.
    """
    claim_id = state.get("claim_id")
    masked_text = state.get("masked_text", "")
    
    logger.info(f"Policy Agent started for Claim ID: {claim_id}")
    
    with claim_tracer.start_as_current_span("policy_agent") as span:
        span.set_attribute("claim_id", claim_id)
        
        if not masked_text or masked_text == "No text extracted from documents.":
            logger.warning("No text found for clause matching. Returning empty clauses.")
            return {"policy_clauses": []}
            
        # 1. Ask LLM to extract retrieval keywords / search queries
        system_prompt = (
            "You are an insurance underwriting assistant. Extract a concise, space-separated list of "
            "search keywords from this claim description that represent: the type of claim, incident details, "
            "diagnoses, treatments, or requested coverages. Return ONLY the space-separated terms, nothing else."
        )
        
        try:
            keywords_query = await llm_service.call_llm(
                prompt=f"{system_prompt}\n\nClaim Text:\n{masked_text[:4000]}",
                max_tokens=100
            )
            keywords_query = keywords_query.strip()
            span.set_attribute("extracted_keywords", keywords_query)
            logger.info(f"Extracted search keywords: '{keywords_query}'")
        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}. Falling back to default slicing.")
            keywords_query = " ".join(masked_text.split()[:20])
            
        # Resolve target policy_id
        target_policy_id = None
        from app.db.models import PolicyDocument
        from app.db.session import AsyncSessionLocal
        
        try:
            async with AsyncSessionLocal() as session:
                res_policy = await session.execute(select(PolicyDocument).limit(1))
                db_policy = res_policy.scalars().first()
                if db_policy:
                    target_policy_id = str(db_policy.id)
        except Exception as db_err:
            logger.error(f"Error accessing DB for target policy: {db_err}")

        # If not in DB, scroll fallback
        if not target_policy_id:
            try:
                from app.rag.vector_store import vector_store_manager
                import asyncio
                res_points, _ = await asyncio.to_thread(
                    vector_store_manager.client.scroll,
                    collection_name=vector_store_manager.collection_name,
                    limit=20,
                    with_payload=True,
                    with_vectors=False
                )
                policy_ids = list(set(point.payload.get("policy_id") for point in res_points if point.payload and "policy_id" in point.payload))
                if policy_ids:
                    target_policy_id = policy_ids[0]
            except Exception as qdrant_err:
                logger.error(f"Error scrolling Qdrant for fallback policy_ids: {qdrant_err}")

        # 2. Query hybrid FAISS + BM25 retriever
        try:
            matched_clauses = await policy_retriever.retrieve_clauses(keywords_query, policy_id=target_policy_id)
            logger.info(f"Retrieved {len(matched_clauses)} matching policy clauses for policy_id: {target_policy_id}")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            matched_clauses = []
            
        # 3. Provider Trust Extraction and Historical Metrics Verification
        company_trust_status = {
            "provider_name": "Unknown",
            "status": "Unverified",
            "trustworthiness": "Unknown",
            "license_id": "N/A",
            "risk_rating": "Unknown",
            "trust_analysis": "No trust information retrieved."
        }
        company_claim_history = {
            "total_claims": 0,
            "approved_claims": 0,
            "rejected_claims": 0,
            "review_claims": 0,
            "avg_fraud_score": 0.0
        }

        try:
            # A. Extract provider name from masked_text using LLM
            extract_prompt = (
                "You are an insurance administration bot.\n"
                "Analyze the claim text and extract the exact name of the primary medical provider, hospital, "
                "or insurance company associated with this claim.\n"
                "Return ONLY the extracted name, nothing else. If not clear, return 'Unknown'."
            )
            provider_name = await llm_service.call_llm(
                prompt=f"{extract_prompt}\n\nClaim Details:\n{masked_text[:2000]}",
                max_tokens=50
            )
            provider_name = provider_name.strip()
            
            if provider_name and provider_name.lower() != "unknown":
                # B. Query Qdrant for company trust registry status (using policy_id = "e4b17188-2884-4974-bf64-ccb360120028")
                matched_trust = await policy_retriever.retrieve_clauses(provider_name, policy_id="e4b17188-2884-4974-bf64-ccb360120028")
                
                if matched_trust:
                    trust_context = "\n\n".join([c["text"] for c in matched_trust])
                    system_prompt_trust = (
                        "You are an insurance compliance officer.\n"
                        "Verify the provider details using the matched Trust Registry clauses.\n"
                        "Return your analysis strictly as a JSON object with fields:\n"
                        "{\n"
                        "  \"provider_name\": \"extracted provider name\",\n"
                        "  \"status\": \"status from registry (e.g. Active, Suspended, Under Review, etc.)\",\n"
                        "  \"trustworthiness\": \"trustworthiness description\",\n"
                        "  \"license_id\": \"license id or 'N/A'\",\n"
                        "  \"risk_rating\": \"Risk Rating (Low/Medium/High/Unknown)\",\n"
                        "  \"trust_analysis\": \"1-2 sentence explanation of your verification\"\n"
                        "}"
                    )
                    trust_response = await llm_service.call_llm(
                        prompt=f"{system_prompt_trust}\n\nRegistry Context:\n{trust_context}\n\nProvider Name: {provider_name}",
                        max_tokens=350
                    )
                    
                    cleaned_trust = trust_response.strip()
                    if cleaned_trust.startswith("```json"):
                        cleaned_trust = cleaned_trust[7:]
                    if cleaned_trust.endswith("```"):
                        cleaned_trust = cleaned_trust[:-3]
                    cleaned_trust = cleaned_trust.strip()
                    
                    import json
                    company_trust_status = json.loads(cleaned_trust)
                
                # C. Query PostgreSQL database to compile historical claim stats for this provider
                from app.db.session import AsyncSessionLocal
                
                async with AsyncSessionLocal() as session:
                    db_query = select(Claim).where(Claim.status == "done")
                    if claim_id:
                        db_query = db_query.where(Claim.id != uuid.UUID(claim_id) if isinstance(claim_id, str) else Claim.id != claim_id)
                    
                    res_claims = await session.execute(db_query)
                    all_claims = res_claims.scalars().all()
                    
                    matching_claims = []
                    for c in all_claims:
                        in_report = provider_name.lower() in (c.report or "").lower()
                        in_match = False
                        if c.policy_match and isinstance(c.policy_match, dict):
                            in_match = provider_name.lower() in str(c.policy_match).lower()
                        
                        if in_report or in_match:
                            matching_claims.append(c)
                            
                    total_c = len(matching_claims)
                    if total_c > 0:
                        approved_c = sum(1 for c in matching_claims if c.decision == "approved")
                        rejected_c = sum(1 for c in matching_claims if c.decision == "rejected")
                        review_c = sum(1 for c in matching_claims if c.decision in ["review", "investigate"])
                        scores = [c.fraud_score for c in matching_claims if c.fraud_score is not None]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        
                        company_claim_history = {
                            "total_claims": total_c,
                            "approved_claims": approved_c,
                            "rejected_claims": rejected_c,
                            "review_claims": review_c,
                            "avg_fraud_score": round(avg_score, 2)
                        }
        except Exception as e:
            logger.error(f"Error compiling company trust/history: {e}")
            
        return {
            "policy_id": target_policy_id or "",
            "policy_clauses": matched_clauses,
            "company_trust_status": company_trust_status,
            "company_claim_history": company_claim_history
        }
