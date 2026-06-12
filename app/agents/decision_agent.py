import logging
from typing import Dict, Any
from app.core.telemetry import claim_tracer

logger = logging.getLogger(__name__)

async def decision_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the final claims decision (approved/rejected/investigate) and confidence
    based on coverage checks, discrepancies, and fraud scores.
    """
    claim_id = state.get("claim_id")
    fraud_score = state.get("fraud_score", 0.0)
    claim_analysis = state.get("claim_analysis", {})
    
    logger.info(f"Decision Agent started for Claim ID: {claim_id}")
    
    with claim_tracer.start_as_current_span("decision_agent") as span:
        span.set_attribute("claim_id", claim_id)
        
        is_covered = claim_analysis.get("is_covered", False)
        discrepancies = claim_analysis.get("discrepancies", [])
        
        # Core Heuristics Logic
        if fraud_score >= 0.7:
            decision = "investigate"
            confidence = float(min(0.5 + (fraud_score / 2.0), 1.0))
            logger.info(f"High fraud score ({fraud_score:.2f}) -> Decision: investigate")
            
        elif fraud_score >= 0.3:
            decision = "investigate"
            confidence = float(0.6 + (0.4 * (fraud_score - 0.3) / 0.4))
            logger.info(f"Moderate fraud score ({fraud_score:.2f}) -> Decision: investigate")
            
        elif not is_covered:
            decision = "rejected"
            confidence = 0.90 if len(discrepancies) > 0 else 0.75
            logger.info("Claim is not covered by policy -> Decision: rejected")
            
        elif len(discrepancies) > 0:
            # Low fraud score, covered, but has some discrepancy
            decision = "investigate"
            confidence = 0.70
            logger.info("Discrepancies found on covered claim -> Decision: investigate")
            
        else:
            # Low fraud, covered, no discrepancies
            decision = "approved"
            confidence = float(1.0 - fraud_score)  # High confidence if fraud score is very close to 0
            logger.info("No risks found -> Decision: approved")
            
        span.set_attribute("final_decision", decision)
        span.set_attribute("confidence", confidence)
        
        return {
            "decision": decision,
            "confidence": confidence
        }
