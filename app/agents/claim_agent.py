import logging
import json
from typing import Dict, Any
from app.core.telemetry import claim_tracer
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

async def claim_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compares the claim text against retrieved policy clauses to analyze coverage validity,
    limits, exclusions, deductibles, and any discrepancies.
    """
    claim_id = state.get("claim_id")
    masked_text = state.get("masked_text", "")
    claim_type = state.get("claim_type", "health")
    policy_clauses = state.get("policy_clauses", [])
    
    logger.info(f"Claim Agent started for Claim ID: {claim_id}")
    
    with claim_tracer.start_as_current_span("claim_agent") as span:
        span.set_attribute("claim_id", claim_id)
        span.set_attribute("policy_clauses_count", len(policy_clauses))
        
        # Prepare context of clauses
        clauses_context = ""
        for i, clause in enumerate(policy_clauses):
            clauses_context += f"Clause {i+1} (Document: {clause.get('doc_name')}):\n{clause.get('text')}\n\n"
            
        system_prompt = (
            "You are an expert insurance claim adjuster. Your task is to evaluate the claim against "
            f"the provided policy clauses, specifically for a claim category of '{claim_type}'.\n"
            "First, verify if the uploaded claim document content matches this category (e.g. a medical invoice "
            "for 'health' claims, a vehicle repair receipt for 'motor' claims, a life insurance proof for 'life' claims, etc.). "
            f"If the document's content does NOT match the category '{claim_type}' (for example, if a motor repair bill is "
            "uploaded under a health claim), you MUST set 'is_covered' to false and add the discrepancy: "
            f"'Document category mismatch: Uploaded document content does not align with the selected claim category {claim_type}.'\n"
            "Second, analyze limits, deductibles, deadlines, and explicit exclusions.\n"
            "You MUST output your response strictly as a JSON object with the following fields:\n"
            "{\n"
            "  \"is_covered\": boolean,\n"
            "  \"coverage_evaluation\": \"detailed text evaluating if the claim fits the policy terms\",\n"
            "  \"matching_clauses_summary\": \"summary of the relevant clauses\",\n"
            "  \"discrepancies\": [\"list of any contradictions, e.g., service date after policy expiration, exclusions, or document mismatch\"],\n"
            "  \"deductible_applied\": float or null,\n"
            "  \"maximum_payout\": float or null\n"
            "}"
        )
        
        user_prompt = f"Policy Clauses:\n{clauses_context}\nClaim Details:\n{masked_text[:4000]}"
        
        try:
            llm_response = await llm_service.call_llm(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                max_tokens=1000
            )
            
            # Clean LLM output in case it includes markdown wrappers
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            claim_analysis = json.loads(cleaned_response)
            logger.info("Claim Agent successfully completed analysis.")
        except Exception as e:
            logger.error(f"Error during claim analysis LLM call or JSON parsing: {e}")
            # Fallback safe analysis dict
            claim_analysis = {
                "is_covered": False,
                "coverage_evaluation": f"Error performing analysis: {str(e)}",
                "matching_clauses_summary": "Failed to extract matching clauses.",
                "discrepancies": ["System processing error occurred during analysis."],
                "deductible_applied": None,
                "maximum_payout": None
            }
            
        return {
            "claim_analysis": claim_analysis
        }
