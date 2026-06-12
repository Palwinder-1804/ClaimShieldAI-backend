import logging
from typing import Dict, Any, List
from app.core.telemetry import claim_tracer
from app.ml.fraud_model import fraud_detector

logger = logging.getLogger(__name__)

async def fraud_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts features from the claim state, evaluates fraud likelihood using ML models
    (XGBoost and Isolation Forest), and identifies primary fraud risk triggers.
    """
    claim_id = state.get("claim_id")
    masked_text = state.get("masked_text", "")
    claim_analysis = state.get("claim_analysis", {})
    
    logger.info(f"Fraud Agent started for Claim ID: {claim_id}")
    
    with claim_tracer.start_as_current_span("fraud_agent") as span:
        span.set_attribute("claim_id", claim_id)
        
        # 1. Feature Extraction
        # We parse the text and analysis details to populate the ML model features.
        # In a real setup, these features would be joined with DB user/policy history.
        # Here we extract them dynamically using text indicators and analysis parameters:
        
        # Extract claimed amount
        claimed_amount = 0.0
        try:
            # Look for numerical patterns or parse from analysis notes
            import re
            payout = claim_analysis.get("maximum_payout")
            if payout:
                claimed_amount = float(payout)
            else:
                matches = re.findall(r"\$\s*([0-9,]+(?:\.[0-9]+)?)", masked_text)
                if matches:
                    claimed_amount = float(matches[0].replace(",", ""))
        except Exception:
            claimed_amount = 500.0  # Default fallback indicator
            
        # Extract weekend indicator
        is_weekend_claim = 1 if any(word in masked_text.lower() for word in ["sunday", "saturday"]) else 0
        
        # Check provider discrepancies
        duplicate_provider = 1 if "duplicate" in masked_text.lower() or "multiple providers" in masked_text.lower() else 0
        
        # Diagnosis-treatment matching
        diagnosis_treatment_match = 1
        if "discrepancies" in claim_analysis and len(claim_analysis["discrepancies"]) > 0:
            diagnosis_treatment_match = 0
            
        # Policy metrics
        policy_age_days = 120  # Default representative average
        provider_age_days = 365  # Default
        days_since_incident = 5  # Default
        amount_to_policy_ratio = min(claimed_amount / 5000.0, 1.5)  # Ratio relative to hypothetical $5000 cap
        num_documents = len(state.get("documents", []))
        
        features = {
            "claimed_amount": claimed_amount,
            "policy_age_days": policy_age_days,
            "num_documents": num_documents,
            "provider_age_days": provider_age_days,
            "days_since_incident": days_since_incident,
            "amount_to_policy_ratio": amount_to_policy_ratio,
            "is_weekend_claim": is_weekend_claim,
            "diagnosis_treatment_match": diagnosis_treatment_match,
            "duplicate_provider": duplicate_provider
        }
        
        logger.info(f"Extracted ML Features: {features}")
        
        # 2. Invoke ML fraud model
        try:
            fraud_results = await fraud_detector.predict(features, claim_id=claim_id)
            fraud_score = fraud_results["fraud_score"]
            fraud_reasons = fraud_results["fraud_reasons"]
            
            logger.info(f"ML Fraud Score: {fraud_score:.4f}, Reasons: {fraud_reasons}")
        except Exception as e:
            logger.error(f"Error executing ML fraud detection: {e}. Falling back to default scoring.")
            # Fallback based on claim discrepancy count
            discrepancy_count = len(claim_analysis.get("discrepancies", []))
            fraud_score = min(0.15 + 0.25 * discrepancy_count, 1.0)
            
            reasons = []
            if discrepancy_count > 0:
                reasons.append(f"Discrepancies flagged in claim content ({discrepancy_count})")
            if is_weekend_claim:
                reasons.append("Claim filed on a weekend (higher anomaly risk)")
            fraud_reasons = reasons if reasons else ["Standard low-risk profile"]
            
        # 3. Incorporate provider trust registry and claims history
        company_trust_status = state.get("company_trust_status", {})
        company_claim_history = state.get("company_claim_history", {})
        
        risk_rating = company_trust_status.get("risk_rating", "Unknown")
        provider_status = company_trust_status.get("status", "Unverified")
        provider_name = company_trust_status.get("provider_name", "Unknown")
        
        if risk_rating == "High" or provider_status == "Suspended":
            fraud_score = max(fraud_score, 0.95)
            fraud_reasons.append(f"Provider '{provider_name}' is listed as SUSPENDED or HIGH RISK in the company registry.")
        elif risk_rating == "Medium" or provider_status == "Under Review":
            fraud_score = max(fraud_score, 0.50)
            if f"Provider '{provider_name}' has warning status (Medium Risk / Under Review) in the registry." not in fraud_reasons:
                fraud_reasons.append(f"Provider '{provider_name}' has warning status (Medium Risk / Under Review) in the registry.")
            
        total_history = company_claim_history.get("total_claims", 0)
        rejected_history = company_claim_history.get("rejected_claims", 0)
        avg_score_history = company_claim_history.get("avg_fraud_score", 0.0)
        
        if total_history >= 2:
            rejection_ratio = rejected_history / total_history
            if rejection_ratio >= 0.5:
                fraud_score = max(fraud_score, 0.75)
                fraud_reasons.append(f"Historical alert: Provider '{provider_name}' has a high claim rejection rate ({rejected_history}/{total_history}).")
            elif avg_score_history >= 0.6:
                fraud_score = max(fraud_score, 0.65)
                fraud_reasons.append(f"Historical alert: Provider '{provider_name}' has a high historical average risk score ({avg_score_history}).")

        span.set_attribute("fraud_score", fraud_score)
        
        return {
            "fraud_score": fraud_score,
            "fraud_reasons": fraud_reasons
        }
