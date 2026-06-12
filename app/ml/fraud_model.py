import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Feature keys used by the models
FEATURE_NAMES = [
    "claimed_amount",
    "policy_age_days",
    "num_documents",
    "provider_age_days",
    "days_since_incident",
    "amount_to_policy_ratio",
    "is_weekend_claim",
    "diagnosis_treatment_match",
    "duplicate_provider"
]

class FraudDetector:
    def __init__(self):
        self.xgb_model = None
        self.isolation_model = None
        self.models_loaded = False
        
        # In production, we'd load models from local serialization or MLflow artifact server.
        # e.g., self.xgb_model = joblib.load('models/xgb_fraud.joblib')
        self._load_models()

    def _load_models(self) -> None:
        """
        Attempts to load the pre-trained ML models from files.
        """
        # Placeholders for loading saved weights
        xgb_path = "./data/models/xgb_model.json"
        iso_path = "./data/models/iso_forest.joblib"
        
        if os.path.exists(xgb_path) and os.path.exists(iso_path):
            try:
                import joblib
                import xgboost as xgb
                
                # Load XGBoost
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(xgb_path)
                
                # Load Isolation Forest
                self.isolation_model = joblib.load(iso_path)
                self.models_loaded = True
                logger.info("Successfully loaded pre-trained XGBoost and Isolation Forest models.")
            except Exception as e:
                logger.error(f"Error loading models: {e}. Falling back to heuristics-based scoring.")
        else:
            logger.info("Pre-trained model files not found. Using heuristic scoring for local execution.")

    async def predict(self, features: Dict[str, Any], claim_id: str = "") -> Dict[str, Any]:
        """
        Predicts fraud probability using loaded ML models.
        Falls back to rule-based heuristic scoring if models are not pre-trained.
        Logs inputs and outputs to MLflow.
        """
        # Validate and prepare features list
        feature_vals = [float(features.get(f, 0)) for f in FEATURE_NAMES]
        
        xgb_prob = 0.0
        iso_flag = 0
        fraud_score = 0.0
        fraud_reasons = []

        if self.models_loaded:
            try:
                # Format to 2D shape for model evaluation
                df_features = pd.DataFrame([feature_vals], columns=FEATURE_NAMES)
                
                # Predict probability from XGBoost
                xgb_prob = float(self.xgb_model.predict_proba(df_features)[0][1])
                
                # Predict anomaly from Isolation Forest (1 = normal, -1 = anomaly)
                iso_pred = int(self.isolation_model.predict(df_features)[0])
                iso_flag = 1 if iso_pred == -1 else 0
                
                # Combined formula: 0.7 * XGBoost + 0.3 * Isolation Forest Anomaly
                fraud_score = 0.7 * xgb_prob + 0.3 * iso_flag
            except Exception as e:
                logger.error(f"Model prediction run error: {e}. Using fallback heuristics.")
                fraud_score, fraud_reasons = self._run_heuristics(features)
        else:
            # Generate heuristic fraud score if ML models are not pre-trained
            fraud_score, fraud_reasons = self._run_heuristics(features)
            xgb_prob = fraud_score * 0.8
            iso_flag = 1 if fraud_score > 0.5 else 0

        # Log details to MLflow if running
        try:
            import mlflow
            # Ensure MLflow is configured
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            mlflow.set_experiment("claimshield-fraud-experiment")
            
            with mlflow.start_run(run_name=f"claim-{claim_id}" if claim_id else None, nested=True):
                # Log inputs
                for f_name, f_val in zip(FEATURE_NAMES, feature_vals):
                    mlflow.log_param(f"feature_{f_name}", f_val)
                # Log outputs
                mlflow.log_metric("fraud_score", fraud_score)
                mlflow.log_metric("xgb_prob", xgb_prob)
                mlflow.log_metric("iso_flag", iso_flag)
                logger.info(f"Logged prediction for claim {claim_id} to MLflow.")
        except Exception as e:
            # Fail silently to avoid breaking the core pipeline when MLflow is not running
            logger.debug(f"MLflow logging bypassed: {e}")

        # Populate fraud reasons based on threshold highlights if not already set
        if not fraud_reasons:
            if features.get("duplicate_provider") == 1:
                fraud_reasons.append("Identified duplicate billing provider credentials.")
            if features.get("diagnosis_treatment_match") == 0:
                fraud_reasons.append("Diagnosis code does not align with standard treatments.")
            if features.get("is_weekend_claim") == 1:
                # Add minor flag
                pass
            if features.get("amount_to_policy_ratio", 0.0) > 1.2:
                fraud_reasons.append(f"Claim amount ratio exceeds standard policy boundaries ({features['amount_to_policy_ratio']:.1f}x limit).")
            if fraud_score > 0.6 and not fraud_reasons:
                fraud_reasons.append("Unusual statistical clusters identified in historical anomaly matches.")

        if not fraud_reasons:
            fraud_reasons.append("Standard low-risk profile.")

        return {
            "fraud_score": float(np.clip(fraud_score, 0.0, 1.0)),
            "fraud_reasons": fraud_reasons
        }

    def _run_heuristics(self, features: Dict[str, Any]) -> tuple[float, List[str]]:
        """
        Deterministic heuristics engine for computing fraud score based on claims flags.
        """
        reasons = []
        base_score = 0.05
        
        if features.get("duplicate_provider") == 1:
            base_score += 0.35
            reasons.append("Duplicate provider invoices submitted in near succession.")
            
        if features.get("diagnosis_treatment_match") == 0:
            base_score += 0.30
            reasons.append("Clinical treatments do not match standard diagnosis guidelines.")
            
        if features.get("is_weekend_claim") == 1:
            base_score += 0.10
            # Minor flag
            
        ratio = features.get("amount_to_policy_ratio", 0.0)
        if ratio > 1.3:
            base_score += 0.20
            reasons.append(f"Claimed amount exceeds typical average boundaries ({ratio:.1f}x standard).")
        elif ratio > 1.0:
            base_score += 0.10
            
        days_since = features.get("days_since_incident", 0)
        if days_since > 90:
            base_score += 0.15
            reasons.append(f"Excessive latency in claim notification ({days_since} days post-incident).")
            
        return base_score, reasons

fraud_detector = FraudDetector()
