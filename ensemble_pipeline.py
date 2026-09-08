"""Phase 6: Stacked Ensemble Meta-Learner Pipeline.

Combines XGBoost + Random Forest + Logistic Regression as Level-0 base
estimators. Their out-of-fold (OOF) predictions are used to train a
Level-1 meta-learner (Logistic Regression) that learns the optimal
blending weights optimized for PR-AUC on the validation set.

Architecture:
    Input Features
        │
        ├── XGBoostFraudModel  ──┐
        ├── RandomForestFraudModel ─┤ → [OOF proba matrix] → MetaLearner (LR)
        └── LogisticRegressionFraudModel ─┘              → Final Fraud Probability

Benefits:
    - Reduces single-model variance (ensemble averaging effect)
    - Combines complementary model biases (linear + trees + boosting)
    - Meta-learner adapts blending weights to PR-AUC objective
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from model_pipeline import (
    BaseFraudEstimator,
    LogisticRegressionFraudModel,
    ModelFactory,
    RandomForestFraudModel,
    XGBoostFraudModel,
)

logger = logging.getLogger(__name__)


@dataclass
class EnsembleReport:
    """Performance report comparing base models vs stacked ensemble."""

    base_model_scores: Dict[str, float]
    ensemble_pr_auc: float
    best_base_model: str
    best_base_pr_auc: float
    ensemble_improvement_pct: float
    meta_learner_coefficients: Dict[str, float]
    n_folds_used: int

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "        STACKED ENSEMBLE PERFORMANCE REPORT         ",
            "=" * 55,
        ]
        for name, score in self.base_model_scores.items():
            lines.append(f"  Base Model [{name:22s}] PR-AUC: {score:.5f}")
        lines.append("-" * 55)
        lines.append(f"  Ensemble  [Stacked Meta-Learner    ] PR-AUC: {self.ensemble_pr_auc:.5f}")
        sign = "+" if self.ensemble_improvement_pct >= 0 else ""
        lines.append(f"  Improvement over best base model: {sign}{self.ensemble_improvement_pct:.2f}%")
        lines.append("-" * 55)
        lines.append("  Meta-Learner Blend Weights:")
        for name, coef in self.meta_learner_coefficients.items():
            lines.append(f"    {name:25s}: {coef:+.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)


class StackedFraudEnsemble:
    """Level-2 Stacking Ensemble for fraud detection.

    Trains multiple diverse base estimators on K-fold cross-validation
    to generate unbiased out-of-fold (OOF) probability predictions,
    then trains a meta-learner to optimally blend them.

    Args:
        base_model_names: List of model type names to use as Level-0 estimators.
        n_folds: Number of stratified cross-validation folds for OOF generation.
        meta_learner_C: Regularization strength for the Logistic Regression meta-learner.
            Lower = more regularized blending (more equal weights).
    """

    def __init__(
        self,
        base_model_names: Optional[List[str]] = None,
        n_folds: int = 5,
        meta_learner_C: float = 0.01,
    ):
        self.base_model_names = base_model_names or ["xgboost", "random_forest", "logistic_regression"]
        self.n_folds = n_folds
        self.meta_learner_C = meta_learner_C

        self.base_models: List[BaseFraudEstimator] = []
        self.meta_learner: Optional[LogisticRegression] = None
        self.is_fitted: bool = False
        self.feature_names_: Optional[List[str]] = None
        self.model_name: str = "StackedEnsemble"

        # Filled after fit
        self._last_report: Optional[EnsembleReport] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "StackedFraudEnsemble":
        """Trains the ensemble using OOF stacking strategy.

        Step 1: K-fold cross-validation on X_train to produce OOF predictions
                from each base model.
        Step 2: Train final base models on the FULL X_train set.
        Step 3: Train meta-learner on the OOF predictions matrix.
        Step 4: Optionally evaluate on validation set and log report.

        Args:
            X_train: Training feature matrix (pre-processed).
            y_train: Training labels.
            X_val: Optional validation set for performance reporting.
            y_val: Optional validation labels.

        Returns:
            self (fitted StackedFraudEnsemble).
        """
        logger.info(
            f"Training StackedEnsemble with {len(self.base_model_names)} base models, "
            f"{self.n_folds} OOF folds..."
        )
        self.feature_names_ = list(X_train.columns)
        n_samples = len(X_train)
        n_base = len(self.base_model_names)

        # --- Step 1: Generate Out-of-Fold predictions ---
        oof_matrix = np.zeros((n_samples, n_base))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            logger.info(f"  OOF Fold {fold_idx + 1}/{self.n_folds}")
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            for model_idx, model_name in enumerate(self.base_model_names):
                fold_model = ModelFactory.create(model_name)
                fold_model.fit(X_fold_train, y_fold_train)
                oof_matrix[val_idx, model_idx] = fold_model.predict_proba(X_fold_val)

        oof_train_pr_aucs: Dict[str, float] = {}
        for model_idx, model_name in enumerate(self.base_model_names):
            score = float(average_precision_score(y_train, oof_matrix[:, model_idx]))
            oof_train_pr_aucs[model_name] = score
            logger.info(f"  OOF PR-AUC [{model_name}]: {score:.5f}")

        # --- Step 2: Train final base models on full training set ---
        logger.info("Training final base models on full training set...")
        self.base_models = []
        for model_name in self.base_model_names:
            final_model = ModelFactory.create(model_name)
            final_model.fit(X_train, y_train)
            self.base_models.append(final_model)

        # --- Step 3: Train meta-learner on OOF predictions ---
        logger.info("Training Logistic Regression meta-learner on OOF predictions...")
        self.meta_learner = LogisticRegression(
            C=self.meta_learner_C,
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )
        self.meta_learner.fit(oof_matrix, y_train)
        self.is_fitted = True

        # --- Step 4: Evaluation Report ---
        meta_coefs = dict(zip(self.base_model_names, self.meta_learner.coef_[0].tolist()))

        ensemble_val_pr_auc = 0.0
        if X_val is not None and y_val is not None:
            val_probas = self.predict_proba(X_val)
            ensemble_val_pr_auc = float(average_precision_score(y_val, val_probas))

            val_base_scores = {}
            for model, name in zip(self.base_models, self.base_model_names):
                val_base_scores[name] = float(average_precision_score(y_val, model.predict_proba(X_val)))

            best_base_name = max(val_base_scores, key=lambda k: val_base_scores[k])
            best_base_score = val_base_scores[best_base_name]
            improvement = 100 * (ensemble_val_pr_auc - best_base_score) / (best_base_score + 1e-9)

            self._last_report = EnsembleReport(
                base_model_scores=val_base_scores,
                ensemble_pr_auc=ensemble_val_pr_auc,
                best_base_model=best_base_name,
                best_base_pr_auc=best_base_score,
                ensemble_improvement_pct=improvement,
                meta_learner_coefficients=meta_coefs,
                n_folds_used=self.n_folds,
            )
            logger.info(f"\n{self._last_report}")
        else:
            # Report on OOF train scores only
            best_base_name = max(oof_train_pr_aucs, key=lambda k: oof_train_pr_aucs[k])
            best_base_score = oof_train_pr_aucs[best_base_name]
            self._last_report = EnsembleReport(
                base_model_scores=oof_train_pr_aucs,
                ensemble_pr_auc=ensemble_val_pr_auc,
                best_base_model=best_base_name,
                best_base_pr_auc=best_base_score,
                ensemble_improvement_pct=0.0,
                meta_learner_coefficients=meta_coefs,
                n_folds_used=self.n_folds,
            )

        return self

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Constructs the meta-feature matrix from base model predictions."""
        if not self.base_models:
            raise RuntimeError("Base models are not fitted. Call fit() first.")
        return np.column_stack([m.predict_proba(X) for m in self.base_models])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns final blended fraud probability from the meta-learner.

        Args:
            X: Pre-processed feature matrix.

        Returns:
            1D array of fraud probabilities in [0, 1].
        """
        if not self.is_fitted or self.meta_learner is None:
            raise RuntimeError("StackedFraudEnsemble must be fitted before calling predict_proba.")
        meta_features = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Returns binary fraud predictions using blended probabilities."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate_pr_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluates ensemble PR-AUC score on a labeled dataset."""
        probas = self.predict_proba(X)
        score = float(average_precision_score(y, probas))
        logger.info(f"[{self.model_name}] Ensemble PR-AUC: {score:.5f}")
        return score

    @property
    def report(self) -> Optional[EnsembleReport]:
        """Returns the latest ensemble performance report."""
        return self._last_report

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the full ensemble (base models + meta-learner) to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Saved StackedFraudEnsemble to {target_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "StackedFraudEnsemble":
        """Deserializes a StackedFraudEnsemble from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, StackedFraudEnsemble):
            raise TypeError(f"Loaded object is not a StackedFraudEnsemble: {type(obj)}")
        return obj
