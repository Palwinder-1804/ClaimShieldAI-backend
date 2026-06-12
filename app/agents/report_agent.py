import logging
from typing import Dict, Any
from app.core.telemetry import claim_tracer
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

async def report_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesizes coverage evaluations, fraud scores, indicators, and policy terms
    into a structured Markdown audit report for adjusters and administrators.
    """
    claim_id = state.get("claim_id")
    fraud_score = state.get("fraud_score", 0.0)
    fraud_reasons = state.get("fraud_reasons", [])
    claim_analysis = state.get("claim_analysis", {})
    policy_clauses = state.get("policy_clauses", [])
    company_trust_status = state.get("company_trust_status", {})
    company_claim_history = state.get("company_claim_history", {})
    
    logger.info(f"Report Agent started for Claim ID: {claim_id}. Fraud Score: {fraud_score}")
    
    with claim_tracer.start_as_current_span("report_agent") as span:
        span.set_attribute("claim_id", claim_id)
        span.set_attribute("fraud_score", fraud_score)
        
        # Determine routing flag
        if fraud_score > 0.7:
            flag = "INVESTIGATE"
        elif fraud_score >= 0.3:
            flag = "REVIEW"
        else:
            flag = "CLEAR"
            
        span.set_attribute("fraud_flag", flag)
        
        # Prepare content context
        reasons_list = "\n".join([f"- {reason}" for reason in fraud_reasons])
        clauses_list = "\n".join([f"- {c.get('doc_name')}: {c.get('text')[:200]}..." for c in policy_clauses])
        
        from datetime import datetime
        report_date = datetime.now().strftime("%B %d, %Y")

        system_prompt = (
            "You are an insurance systems reporting expert. Create a comprehensive, formal, "
            "and highly structured claim investigation audit report in Markdown format.\n"
            "Do NOT wrap the response in a markdown code block (e.g. do NOT start with ```markdown and end with ```). "
            "Output the raw Markdown directly.\n"
            "Include sections for: Executive Summary, Coverage Assessment, Provider Trust & History, Fraud & Risk Analysis, "
            "Policy Mapping, and Final Processing Recommendation.\n"
            "Maintain an objective, technical tone, and reference specific scores and reasons.\n"
            "At the very end of the report, include the following signature block exactly (do not use placeholders):\n"
            "**Prepared by:**  \n"
            "InsuranceAI  \n"
            "Insurance Systems Reporting Expert  \n"
            f"{report_date}"
        )
        
        user_prompt = (
            f"Claim ID: {claim_id}\n"
            f"Fraud Flag: {flag} (Score: {fraud_score:.2f})\n"
            f"Fraud Red Flags Identified:\n{reasons_list}\n\n"
            f"Coverage Analysis Result:\n{claim_analysis.get('coverage_evaluation', 'N/A')}\n"
            f"Discrepancies found: {claim_analysis.get('discrepancies', [])}\n"
            f"Deductible applied: {claim_analysis.get('deductible_applied')}\n"
            f"Maximum payout: {claim_analysis.get('maximum_payout')}\n\n"
            f"Provider Trust Status details:\n"
            f"- Provider Name: {company_trust_status.get('provider_name')}\n"
            f"- Status: {company_trust_status.get('status')}\n"
            f"- Trustworthiness: {company_trust_status.get('trustworthiness')}\n"
            f"- License ID: {company_trust_status.get('license_id')}\n"
            f"- Risk Rating: {company_trust_status.get('risk_rating')}\n"
            f"- Trust Analysis: {company_trust_status.get('trust_analysis')}\n\n"
            f"Provider Historical Claims Metrics (from DB):\n"
            f"- Total Claims Processed: {company_claim_history.get('total_claims')}\n"
            f"- Approved Claims: {company_claim_history.get('approved_claims')}\n"
            f"- Rejected Claims: {company_claim_history.get('rejected_claims')}\n"
            f"- Review/Investigate Claims: {company_claim_history.get('review_claims')}\n"
            f"- Average Risk Score: {company_claim_history.get('avg_fraud_score')}\n\n"
            f"Policy clauses checked:\n{clauses_list}"
        )
        
        try:
            report_text = await llm_service.call_llm(
                prompt=f"{system_prompt}\n\n{user_prompt}",
                max_tokens=1500
            )
            logger.info("Report Agent successfully compiled report.")
        except Exception as e:
            logger.error(f"Error compiling report: {e}")
            report_text = (
                f"# Claim Audit Report (System Error Fallback)\n\n"
                f"**Claim ID:** {claim_id}\n"
                f"**System Warning:** Report compiler encountered an error: {str(e)}\n\n"
                f"### System Metrics:\n"
                f"- Fraud Score: {fraud_score:.2f} ({flag})\n"
                f"- Covered: {claim_analysis.get('is_covered')}\n"
            )
            
        return {
            "report": report_text
        }
