"""Production FastAPI Real-Time Fraud Scoring & Explainability Microservice.

Provides low-latency transaction inference, SHAP-based local explainability reason codes,
cost-aware decision routing (Approve, Review, Decline), and streaming online learning endpoints.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config import CONFIG
from data_preprocessing import LeakageSafePreprocessor, generate_synthetic_fraud_dataset
from explainability import FeatureContribution, FraudExplainer
from model_pipeline import BaseFraudEstimator
from online_learning import OnlineFraudLearner

logger = logging.getLogger("FraudAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TransactionPayload(BaseModel):
    """Input payload representing a single credit card transaction event."""

    transaction_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transaction identifier",
    )
    Time: float = Field(default=0.0, description="Elapsed seconds since reference epoch")
    Amount: float = Field(default=50.0, gt=0.0, description="Transaction monetary amount in USD")
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)


class PredictionResponse(BaseModel):
    """Scoring response containing probability, action, threshold, and SHAP reason codes."""

    transaction_id: str
    fraud_probability: float
    decision: str  # 'APPROVE', 'MANUAL_REVIEW', 'DECLINE'
    is_fraud: bool
    decision_threshold: float
    expected_financial_loss: float
    top_risk_reasons: List[Dict[str, Any]]


class DetailedExplainResponse(BaseModel):
    """Full explainability breakdown including base value and all SHAP feature values."""

    transaction_id: str
    fraud_probability: float
    base_value: float
    top_features: List[Dict[str, Any]]
    all_feature_attributions: Dict[str, float]


class FeedbackPayload(BaseModel):
    """Transaction outcome feedback for streaming online incremental learning."""

    transaction: TransactionPayload
    is_fraud_confirmed: int = Field(..., ge=0, le=1, description="1 if confirmed fraud, 0 if legitimate")


class OnlineUpdateResponse(BaseModel):
    """Telemetry response following an incremental online model update."""

    status: str
    samples_ingested: int
    frauds_learned: int
    pre_update_log_loss: float
    post_update_log_loss: float
    running_average_loss: float
    cumulative_samples_seen: int


# ---------------------------------------------------------------------------
# Global State Container & Lifespan Loader
# ---------------------------------------------------------------------------

class ServiceState:
    model: Optional[BaseFraudEstimator] = None
    preprocessor: Optional[LeakageSafePreprocessor] = None
    explainer: Optional[FraudExplainer] = None
    online_learner: Optional[OnlineFraudLearner] = None
    threshold_metadata: Dict[str, Any] = {}
    optimal_threshold: float = 0.50


STATE = ServiceState()


def load_or_bootstrap_artifacts() -> None:
    """Loads trained artifacts from disk or executes training pipeline if missing."""
    model_path = CONFIG.artifact_dir / CONFIG.model_artifact_name
    preproc_path = CONFIG.artifact_dir / CONFIG.preprocessor_artifact_name
    thresh_path = CONFIG.artifact_dir / CONFIG.threshold_artifact_name
    online_path = CONFIG.artifact_dir / CONFIG.online_model_artifact_name

    if not (model_path.exists() and preproc_path.exists() and thresh_path.exists()):
        logger.warning("Required model artifacts not found. Bootstrapping initial training pipeline...")
        from train import run_training_pipeline
        run_training_pipeline(model_type="xgboost", imbalance_strategy="hybrid", n_samples=10000)

    logger.info("Loading production artifacts into memory...")
    STATE.model = BaseFraudEstimator.load(model_path)
    STATE.preprocessor = joblib.load(preproc_path)

    with open(thresh_path, "r") as f:
        STATE.threshold_metadata = json.load(f)
    STATE.optimal_threshold = STATE.threshold_metadata.get("optimal_threshold", 0.50)

    if online_path.exists():
        STATE.online_learner = OnlineFraudLearner.load(online_path)
    else:
        STATE.online_learner = OnlineFraudLearner()

    # Initialize SHAP explainer
    logger.info("Initializing SHAP Explainer...")
    # Generate small background dataset for explainer baseline
    sample_df = generate_synthetic_fraud_dataset(n_samples=200, random_state=42)
    sample_features = sample_df.drop(columns=["Class"])
    bg_transformed = STATE.preprocessor.transform(sample_features)
    STATE.explainer = FraudExplainer(model_wrapper=STATE.model, background_data=bg_transformed)
    logger.info(f"Microservice initialized. Decision Threshold: {STATE.optimal_threshold:.4f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook handling startup loading and graceful shutdown."""
    load_or_bootstrap_artifacts()
    yield
    logger.info("Shutting down Fraud Detection Microservice...")


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection & Real-Time Scoring API",
    description="Production-grade ML inference engine with PR-AUC optimization, cost-aware thresholds, SHAP explainability, and streaming online learning.",
    version="1.0.0",
    lifespan=lifespan,
)


def _payload_to_dataframe(payload: TransactionPayload) -> pd.DataFrame:
    """Converts a Pydantic transaction payload into a single-row DataFrame."""
    data_dict = payload.model_dump()
    data_dict.pop("transaction_id", None)
    return pd.DataFrame([data_dict])


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, Any]:
    """Returns service health, loaded model metadata, and operational threshold."""
    return {
        "status": "HEALTHY",
        "model_loaded": STATE.model is not None,
        "model_name": STATE.model.model_name if STATE.model else None,
        "optimal_threshold": STATE.optimal_threshold,
        "pr_auc": STATE.threshold_metadata.get("pr_auc"),
        "max_review_rate_cap": STATE.threshold_metadata.get("max_review_rate_cap"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(payload: TransactionPayload) -> PredictionResponse:
    """Evaluates a single transaction in real time.

    Returns:
        - Fraud probability score
        - Business action (APPROVE, MANUAL_REVIEW, DECLINE)
        - Expected financial loss
        - Top SHAP risk factors with human-readable reason codes
    """
    if STATE.model is None or STATE.preprocessor is None or STATE.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently initializing or unavailable.",
        )

    # 1. Preprocess raw transaction data
    df_raw = _payload_to_dataframe(payload)
    df_trans = STATE.preprocessor.transform(df_raw)

    # 2. Score probability
    fraud_prob = float(STATE.model.predict_proba(df_trans)[0])
    threshold = STATE.optimal_threshold

    # 3. Apply 3-Tier Business Routing Logic
    #   - Prob < 0.5 * Threshold -> Instant APPROVE
    #   - 0.5 * Threshold <= Prob < Threshold -> Low-friction APPROVE / Step-up
    #   - Threshold <= Prob < 0.85 -> MANUAL_REVIEW (Alert queue)
    #   - Prob >= 0.85 -> Immediate DECLINE
    is_flagged = fraud_prob >= threshold

    if fraud_prob >= 0.85:
        decision = "DECLINE"
    elif fraud_prob >= threshold:
        decision = "MANUAL_REVIEW"
    else:
        decision = "APPROVE"

    # Expected financial loss = P(Fraud) * Cost_FN + (1-P(Fraud)) * Cost_TN
    expected_loss = fraud_prob * CONFIG.cost.cost_false_negative

    # 4. Generate SHAP top feature attributions & reason codes
    explanation = STATE.explainer.explain_transaction(df_trans, top_k=5)
    top_reasons = [
        {
            "rank": c.importance_rank,
            "feature": c.feature_name,
            "shap_value": round(c.shap_value, 4),
            "impact": c.direction,
            "reason": c.reason_code,
        }
        for c in explanation.top_contributions
    ]

    return PredictionResponse(
        transaction_id=payload.transaction_id or str(uuid.uuid4()),
        fraud_probability=round(fraud_prob, 5),
        decision=decision,
        is_fraud=is_flagged,
        decision_threshold=round(threshold, 4),
        expected_financial_loss=round(expected_loss, 2),
        top_risk_reasons=top_reasons,
    )


@app.post("/explain", response_model=DetailedExplainResponse, tags=["Explainability"])
def explain_transaction(payload: TransactionPayload) -> DetailedExplainResponse:
    """Returns comprehensive SHAP decomposition including all feature attributions."""
    if STATE.model is None or STATE.preprocessor is None or STATE.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently initializing or unavailable.",
        )

    df_raw = _payload_to_dataframe(payload)
    df_trans = STATE.preprocessor.transform(df_raw)
    explanation = STATE.explainer.explain_transaction(df_trans, top_k=10)

    top_reasons = [
        {
            "rank": c.importance_rank,
            "feature": c.feature_name,
            "value": round(c.feature_value, 4),
            "shap_value": round(c.shap_value, 4),
            "impact": c.direction,
            "reason": c.reason_code,
        }
        for c in explanation.top_contributions
    ]

    return DetailedExplainResponse(
        transaction_id=payload.transaction_id or str(uuid.uuid4()),
        fraud_probability=round(explanation.predicted_probability, 5),
        base_value=round(explanation.base_value, 5),
        top_features=top_reasons,
        all_feature_attributions={k: round(v, 5) for k, v in explanation.all_shap_values.items()},
    )


@app.post("/feedback", response_model=OnlineUpdateResponse, tags=["Online Learning"])
def ingest_feedback(feedback: FeedbackPayload) -> OnlineUpdateResponse:
    """Receives verified transaction ground truth and performs incremental online learning.

    Enables near real-time adaptation to emerging fraud patterns.
    """
    if STATE.online_learner is None or STATE.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Online learner is not initialized.",
        )

    df_raw = _payload_to_dataframe(feedback.transaction)
    df_trans = STATE.preprocessor.transform(df_raw)

    result = STATE.online_learner.update_incremental(
        X_new=df_trans,
        y_new=[feedback.is_fraud_confirmed],
    )

    # Persist updated weights asynchronously / on checkpoint
    online_path = CONFIG.artifact_dir / CONFIG.online_model_artifact_name
    STATE.online_learner.save(online_path)

    return OnlineUpdateResponse(**result)


@app.get("/metrics", tags=["Monitoring"])
def get_metrics() -> Dict[str, Any]:
    """Returns full model performance scorecard, financial cost matrix, and streaming stats."""
    return {
        "model_type": STATE.model.model_name if STATE.model else None,
        "performance_metrics": STATE.threshold_metadata,
        "online_learning_telemetry": {
            "total_samples_ingested": STATE.online_learner.stats.total_samples_ingested if STATE.online_learner else 0,
            "total_frauds_learned": STATE.online_learner.stats.total_frauds_learned if STATE.online_learner else 0,
            "running_average_loss": STATE.online_learner.stats.running_loss if STATE.online_learner else 0.0,
        },
    }
