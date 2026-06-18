import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END

# Import agent nodes
from app.agents.ingest_agent import ingest_agent
from app.agents.policy_agent import policy_agent
from app.agents.claim_agent import claim_agent
from app.agents.fraud_agent import fraud_agent
from app.agents.report_agent import report_agent
from app.agents.decision_agent import decision_agent

logger = logging.getLogger(__name__)

# 1. Define ClaimState TypedDict
class ClaimState(TypedDict):
    claim_id: str
    documents: List[str]          # MinIO document paths
    claim_type: str               # Selected category (health, motor, life, property)
    extracted_text: str           # Raw extracted text
    masked_text: str              # PII-anonymized text
    policy_id: str                # Matched Policy Document ID
    policy_clauses: List[dict]    # RAG policy matches
    claim_analysis: dict          # Coverage assessment
    company_trust_status: dict    # RAG company trust registry status
    company_claim_history: dict   # Historical claims details
    fraud_score: float            # Combined ML fraud score (0-1)
    fraud_reasons: List[str]      # Fraud risk flags
    report: str                   # Generated audit report text (Markdown)
    decision: str                 # approved/rejected/investigate
    confidence: float             # Decision confidence (0-1)

# 2. Build the LangGraph StateMachine
workflow = StateGraph(ClaimState)

# Add nodes to graph
workflow.add_node("ingest", ingest_agent)
workflow.add_node("policy", policy_agent)
workflow.add_node("claim", claim_agent)
workflow.add_node("fraud", fraud_agent)
workflow.add_node("report", report_agent)
workflow.add_node("decision", decision_agent)

# Configure transitions
workflow.set_entry_point("ingest")
workflow.add_edge("ingest", "policy")
workflow.add_edge("policy", "claim")
workflow.add_edge("claim", "fraud")
workflow.add_edge("fraud", "report")
workflow.add_edge("report", "decision")
workflow.add_edge("decision", END)

# Compile graph
claim_graph = workflow.compile()

class ClaimPipeline:
    @staticmethod
    async def process_claim(claim_id: str, documents: List[str], claim_type: str) -> Dict[str, Any]:
        """
        Runs the full LangGraph claim pipeline asynchronously.
        Tags runs for observability in LangSmith.
        """
        initial_state = {
            "claim_id": claim_id,
            "documents": documents,
            "claim_type": claim_type,
            "extracted_text": "",
            "masked_text": "",
            "policy_id": "",
            "policy_clauses": [],
            "claim_analysis": {},
            "company_trust_status": {},
            "company_claim_history": {},
            "fraud_score": 0.0,
            "fraud_reasons": [],
            "report": "",
            "decision": "pending",
            "confidence": 0.0
        }
        
        # Setup LangSmith tracing context tags
        config = {
            "metadata": {
                "project": "claimshield-ai",
                "claim_id": claim_id
            }
        }
        
        logger.info(f"Initiating LangGraph processing workflow for Claim: {claim_id}")
        
        try:
            final_state = await claim_graph.ainvoke(initial_state, config=config)
            logger.info(f"LangGraph processing workflow completed for Claim: {claim_id}")
            return final_state
        except Exception as e:
            logger.error(f"LangGraph processing failed for Claim {claim_id}: {e}")
            raise
