"""Model Explainability Module using SHAP (SHapley Additive exPlanations).

Provides local per-transaction explainability, generating top contributing
risk factors and reason codes for why a transaction was flagged or approved.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap

from model_pipeline import BaseFraudEstimator, XGBoostFraudModel, RandomForestFraudModel

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Individual feature contribution to a transaction's fraud score."""

    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # 'INCREASES_FRAUD_RISK' or 'DECREASES_FRAUD_RISK'
    importance_rank: int
    reason_code: str


@dataclass
class TransactionExplanation:
    """Full explainability payload for a single scored transaction."""

    base_value: float
    predicted_probability: float
    top_contributions: List[FeatureContribution]
    all_shap_values: Dict[str, float]


class FraudExplainer:
    """Wraps SHAP explainers to provide fast, interpretable reason codes for fraud decisions."""

    # Map PCA/raw features to business-friendly explanation codes
    FEATURE_REASON_MAP: Dict[str, str] = {
        "V1": "Account velocity and location variance",
        "V2": "Transaction frequency anomaly",
        "V3": "Terminal device fingerprint deviation",
        "V4": "High-risk merchant category profile",
        "V7": "Cross-border transaction behavior",
        "V10": "Unusual user spending pattern profile",
        "V11": "Rapid consecutive authorizations",
        "V12": "High-risk IP address / ASN clustering",
        "V14": "Strong behavioral anomaly pattern (Key Risk Driver)",
        "V17": "Critical historical fraud correlation signal",
        "scaled_Amount": "Transaction amount deviation from cardholder baseline",
        "scaled_log_amount": "Extreme transaction value relative to normal usage",
        "hour_sin": "Off-hours transaction timing (Cyclical Sine)",
        "hour_cos": "Off-hours transaction timing (Cyclical Cosine)",
        "scaled_time_hour": "Unusual time of day for cardholder",
    }

    def __init__(self, model_wrapper: BaseFraudEstimator, background_data: Optional[pd.DataFrame] = None):
        self.model_wrapper = model_wrapper
        self.background_data = background_data
        self.explainer: Optional[Any] = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Instantiates appropriate SHAP explainer based on model architecture."""
        estimator = self.model_wrapper.estimator

        if isinstance(self.model_wrapper, (XGBoostFraudModel, RandomForestFraudModel)):
            try:
                # TreeExplainer is fastest and exact for tree ensembles
                self.explainer = shap.TreeExplainer(estimator)
                logger.info("Initialized TreeExplainer for tree-based fraud model.")
            except Exception as e:
                logger.warning(f"TreeExplainer initialization failed ({e}). Falling back to Explainer.")
                bg = self.background_data.sample(min(100, len(self.background_data))) if self.background_data is not None else None
                self.explainer = shap.Explainer(estimator, bg)
        else:
            # Linear / general explainer fallback
            bg = self.background_data.sample(min(100, len(self.background_data))) if self.background_data is not None else None
            self.explainer = shap.Explainer(estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator, bg)
            logger.info("Initialized generic SHAP Explainer.")

    def explain_transaction(
        self,
        X_single: pd.DataFrame,
        top_k: int = 5,
    ) -> TransactionExplanation:
        """Generates SHAP attribution values and top-K ranked risk factors for one transaction.

        Args:
            X_single: Single-row DataFrame matching preprocessed feature schema.
            top_k: Number of most significant contributing features to return.

        Returns:
            TransactionExplanation object containing ranked contributions.
        """
        if len(X_single) != 1:
            raise ValueError(f"explain_transaction expects a single-row DataFrame, got {len(X_single)} rows.")

        probas = self.model_wrapper.predict_proba(X_single)
        fraud_prob = float(probas[0])

        shap_values_raw = self.explainer(X_single)
        
        # Handle different SHAP output formats (binary classification multi-output vs single output)
        if hasattr(shap_values_raw, "values"):
            vals = shap_values_raw.values
            if vals.ndim == 3:  # (1, n_features, 2 classes)
                shap_vec = vals[0, :, 1]
            elif vals.ndim == 2:  # (1, n_features)
                shap_vec = vals[0, :]
            else:
                shap_vec = vals.flatten()
            base_val = float(shap_values_raw.base_values[0, 1] if shap_values_raw.base_values.ndim > 1 else shap_values_raw.base_values[0])
        else:
            shap_vec = np.asarray(shap_values_raw).flatten()
            base_val = 0.0

        feature_names = list(X_single.columns)
        all_shap_dict = {f: float(v) for f, v in zip(feature_names, shap_vec)}

        # Rank features by absolute magnitude of impact
        sorted_indices = np.argsort(np.abs(shap_vec))[::-1]

        contributions: List[FeatureContribution] = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            fname = feature_names[idx]
            fval = float(X_single.iloc[0, idx])
            sval = float(shap_vec[idx])
            direction = "INCREASES_FRAUD_RISK" if sval > 0 else "DECREASES_FRAUD_RISK"
            reason = self.FEATURE_REASON_MAP.get(fname, f"Signal variation in indicator {fname}")

            contributions.append(
                FeatureContribution(
                    feature_name=fname,
                    feature_value=fval,
                    shap_value=sval,
                    direction=direction,
                    importance_rank=rank,
                    reason_code=reason,
                )
            )

        return TransactionExplanation(
            base_value=base_val,
            predicted_probability=fraud_prob,
            top_contributions=contributions,
            all_shap_values=all_shap_dict,
        )
