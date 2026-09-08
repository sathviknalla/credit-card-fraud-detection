"""Phase 5: Probability Calibration Layer.

XGBoost and tree-based models often produce overconfident probabilities.
This module wraps any BaseFraudEstimator with Platt Scaling or Isotonic
Regression post-hoc calibration to produce well-calibrated fraud probabilities
that accurately reflect the true likelihood of fraud.

Usage:
    calibrated_model = CalibrationWrapper(base_model, method="isotonic")
    calibrated_model.fit_calibration(X_cal, y_cal)
    probas = calibrated_model.predict_proba(X_test)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from model_pipeline import BaseFraudEstimator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Summary statistics of model calibration quality."""

    method: str
    brier_score_before: float
    brier_score_after: float
    improvement_pct: float
    mean_calibration_error: float
    n_calibration_samples: int
    is_improved: bool = field(init=False)

    def __post_init__(self):
        self.is_improved = self.brier_score_after < self.brier_score_before

    def __str__(self) -> str:
        status = "✅ IMPROVED" if self.is_improved else "⚠️ NO IMPROVEMENT"
        return (
            f"[{status}] Calibration Method: {self.method}\n"
            f"  Brier Score Before : {self.brier_score_before:.5f}\n"
            f"  Brier Score After  : {self.brier_score_after:.5f}\n"
            f"  Improvement        : {self.improvement_pct:+.2f}%\n"
            f"  Mean Calib. Error  : {self.mean_calibration_error:.5f}\n"
            f"  Calibration Samples: {self.n_calibration_samples:,}"
        )


class _SKLearnEstimatorShim(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible shim wrapping a BaseFraudEstimator.

    CalibratedClassifierCV expects standard sklearn API (fit / predict_proba
    returning [P(0), P(1)] stacked). This adapter bridges the gap.
    """

    def __init__(self, base_model: BaseFraudEstimator):
        self._model = base_model
        self.classes_ = np.array([0, 1])
        
    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y):
        # Already fitted — no-op shim
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        p1 = self._model.predict_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn introspection compatibility
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class CalibrationWrapper:
    """Post-hoc calibration wrapper for any BaseFraudEstimator.

    Applies Platt Scaling (sigmoid) or Isotonic Regression to correct
    overconfident or underconfident raw model probabilities. Uses a held-out
    calibration set to fit the recalibration layer without retraining the base.

    Args:
        base_model: A fitted BaseFraudEstimator instance.
        method: Calibration method. 'sigmoid' = Platt Scaling (better for
            small datasets), 'isotonic' = Isotonic Regression (better for
            large datasets, more flexible but can overfit).
    """

    VALID_METHODS = ("sigmoid", "isotonic")

    def __init__(
        self,
        base_model: BaseFraudEstimator,
        method: Literal["sigmoid", "isotonic"] = "isotonic",
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")

        self.method = method
        self.base_model = base_model
        self.model_name = f"{base_model.model_name}+Calibrated[{method}]"
        self._calibrated_estimator: Optional[CalibratedClassifierCV] = None
        self._is_calibrated: bool = False
        self._shim = _SKLearnEstimatorShim(base_model)

    def fit_calibration(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> CalibrationReport:
        """Fits the calibration layer on a held-out calibration set.

        Args:
            X_cal: Feature matrix for calibration (pre-processed, NOT used to train base model).
            y_cal: Ground truth labels for calibration set.

        Returns:
            CalibrationReport with before/after Brier scores and quality metrics.
        """
        logger.info(
            f"Fitting {self.method} calibration on {len(X_cal):,} samples..."
        )

        # Score BEFORE calibration
        raw_probas = self.base_model.predict_proba(X_cal)
        brier_before = float(brier_score_loss(y_cal, raw_probas))

        try:
            from sklearn.frozen import FrozenEstimator
            estimator_arg = FrozenEstimator(self._shim)
            cv_arg = None
        except ImportError:
            estimator_arg = self._shim
            cv_arg = "prefit"

        # Fit calibration using prefit=True (base model already trained)
        self._calibrated_estimator = CalibratedClassifierCV(
            estimator=estimator_arg,
            method=self.method,
            cv=cv_arg,
        )
        self._calibrated_estimator.fit(X_cal, y_cal)
        self._is_calibrated = True

        # Score AFTER calibration
        cal_probas = self._calibrated_estimator.predict_proba(X_cal)[:, 1]
        brier_after = float(brier_score_loss(y_cal, cal_probas))

        # Mean calibration error via calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_cal, cal_probas, n_bins=10, strategy="uniform"
        )
        mce = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))

        improvement_pct = (
            100 * (brier_before - brier_after) / (brier_before + 1e-9)
        )

        report = CalibrationReport(
            method=self.method,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            improvement_pct=improvement_pct,
            mean_calibration_error=mce,
            n_calibration_samples=len(X_cal),
        )

        logger.info(f"\n{report}")
        return report

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated fraud probabilities for the positive (fraud) class."""
        if not self._is_calibrated or self._calibrated_estimator is None:
            logger.warning("Calibration not fitted. Falling back to raw model probabilities.")
            return self.base_model.predict_proba(X)

        return self._calibrated_estimator.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions using calibrated probabilities."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_calibration_curve(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        save_path: Optional[Union[str, Path]] = None,
        n_bins: int = 10,
    ) -> plt.Figure:
        """Generates a reliability diagram comparing raw vs calibrated probabilities.

        Args:
            X_eval: Feature matrix for evaluation.
            y_eval: True labels.
            save_path: If provided, saves the figure to disk.
            n_bins: Number of probability bins for the calibration curve.

        Returns:
            matplotlib Figure object.
        """
        raw_probas = self.base_model.predict_proba(X_eval)
        cal_probas = self.predict_proba(X_eval)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Calibration Analysis: {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )

        # --- Reliability Diagram ---
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)

        fop_raw, mpv_raw = calibration_curve(y_eval, raw_probas, n_bins=n_bins, strategy="uniform")
        ax.plot(mpv_raw, fop_raw, "o-", color="tomato", label=f"Raw ({self.base_model.model_name})")

        fop_cal, mpv_cal = calibration_curve(y_eval, cal_probas, n_bins=n_bins, strategy="uniform")
        ax.plot(mpv_cal, fop_cal, "s-", color="steelblue", label=f"Calibrated ({self.method})")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Reliability Diagram")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Probability Distribution Histogram ---
        ax2 = axes[1]
        ax2.hist(raw_probas, bins=30, alpha=0.5, color="tomato", label="Raw Probabilities", density=True)
        ax2.hist(cal_probas, bins=30, alpha=0.5, color="steelblue", label="Calibrated Probabilities", density=True)
        ax2.set_xlabel("Predicted Fraud Probability")
        ax2.set_ylabel("Density")
        ax2.set_title("Probability Distribution Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        brier_raw = float(brier_score_loss(y_eval, raw_probas))
        brier_cal = float(brier_score_loss(y_eval, cal_probas))
        fig.text(
            0.5, -0.02,
            f"Brier Score → Raw: {brier_raw:.5f}  |  Calibrated: {brier_cal:.5f}  |  Δ = {brier_raw - brier_cal:+.5f}",
            ha="center",
            fontsize=11,
            color="darkgreen" if brier_cal < brier_raw else "crimson",
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Calibration curve saved to {save_path}")

        return fig

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the calibration wrapper (including base model) to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Saved calibrated model to {target_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CalibrationWrapper":
        """Deserializes a CalibrationWrapper from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, CalibrationWrapper):
            raise TypeError(f"Loaded object is not a CalibrationWrapper: {type(obj)}")
        return obj
