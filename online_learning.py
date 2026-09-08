"""Online & Incremental Learning Module for Streaming Fraud Detection.

Enables the fraud detection system to adapt continuously to newly labeled
chargebacks and confirmed transactions in near real-time via mini-batch partial fitting.
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class StreamingStats:
    """Telemetry tracking for the streaming online learning pipeline."""

    total_samples_ingested: int = 0
    total_frauds_learned: int = 0
    total_legit_learned: int = 0
    recent_losses: List[float] = field(default_factory=list)
    running_loss: float = 0.0
    last_update_timestamp: Optional[str] = None


class OnlineFraudLearner:
    """Thread-safe incremental learner leveraging stochastic gradient descent with log loss.

    Allows fast micro-updates upon feedback arrival (e.g., chargeback notifications,
    analyst verification) without taking the primary model offline.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = CONFIG.models.online_sgd_params.copy()
        if params:
            self.params.update(params)

        self.model = SGDClassifier(
            loss=self.params.get("loss", "log_loss"),
            penalty=self.params.get("penalty", "l2"),
            alpha=self.params.get("alpha", 1e-4),
            learning_rate=self.params.get("learning_rate", "optimal"),
            random_state=self.params.get("random_state", 42),
            warm_start=True,
        )
        self.classes = np.array([0, 1])
        self.is_initialized = False
        self.stats = StreamingStats()
        self._lock = threading.Lock()
        self.feature_names: List[str] = []

    def __getstate__(self) -> Dict[str, Any]:
        """Custom state extraction excluding unpicklable thread Lock."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restores state and reinitializes a fresh thread Lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def initialize_from_batch(
        self,
        X_init: pd.DataFrame,
        y_init: pd.Series,
        epochs: int = 5,
    ) -> "OnlineFraudLearner":
        """Initializes online learner with baseline historical data using partial_fit."""
        with self._lock:
            self.feature_names = list(X_init.columns)
            X_mat = X_init.values
            y_arr = y_init.values.astype(int)

            for epoch in range(epochs):
                self.model.partial_fit(X_mat, y_arr, classes=self.classes)

            self.is_initialized = True
            self.stats.total_samples_ingested = len(y_arr)
            self.stats.total_frauds_learned = int((y_arr == 1).sum())
            self.stats.total_legit_learned = int((y_arr == 0).sum())

            # Evaluate initial baseline loss
            probas = self.model.predict_proba(X_mat)[:, 1]
            loss = float(log_loss(y_arr, probas, labels=[0, 1]))
            self.stats.running_loss = loss
            logger.info(
                f"Online model initialized on {len(y_arr):,} baseline samples. Initial LogLoss: {loss:.4f}"
            )

        return self

    def update_incremental(
        self,
        X_new: pd.DataFrame,
        y_new: Union[pd.Series, List[int], np.ndarray],
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Performs incremental online learning step on a streaming batch of newly labeled samples.

        Args:
            X_new: Feature matrix for newly verified transactions.
            y_new: Ground truth binary labels (0=Legitimate, 1=Confirmed Fraud).
            sample_weights: Optional weights (e.g., giving higher weight to verified frauds).

        Returns:
            Dictionary reporting update status, batch metrics, and streaming telemetry.
        """
        if not self.is_initialized:
            raise RuntimeError("OnlineFraudLearner must be initialized before calling update_incremental().")

        with self._lock:
            X_mat = X_new.values
            y_arr = np.asarray(y_new).astype(int)
            n_samples = len(y_arr)

            # Assign higher loss gradient weight to rare fraud feedback if not provided
            if sample_weights is None:
                sample_weights = np.where(y_arr == 1, 10.0, 1.0)

            # Predict before update to assess pre-update streaming accuracy
            pre_probas = self.model.predict_proba(X_mat)[:, 1]
            pre_loss = float(log_loss(y_arr, pre_probas, labels=[0, 1]))

            # Incremental partial_fit update
            self.model.partial_fit(
                X_mat,
                y_arr,
                classes=self.classes,
                sample_weight=sample_weights,
            )

            # Post-update predictions
            post_probas = self.model.predict_proba(X_mat)[:, 1]
            post_loss = float(log_loss(y_arr, post_probas, labels=[0, 1]))

            # Update telemetry
            n_fraud = int((y_arr == 1).sum())
            n_legit = n_samples - n_fraud

            self.stats.total_samples_ingested += n_samples
            self.stats.total_frauds_learned += n_fraud
            self.stats.total_legit_learned += n_legit
            self.stats.recent_losses.append(post_loss)
            if len(self.stats.recent_losses) > 50:
                self.stats.recent_losses.pop(0)

            self.stats.running_loss = float(np.mean(self.stats.recent_losses))

            logger.info(
                f"[Streaming Update] Ingested {n_samples} txs ({n_fraud} fraud, {n_legit} legit). "
                f"LogLoss improved from {pre_loss:.4f} -> {post_loss:.4f}"
            )

            return {
                "status": "SUCCESS",
                "samples_ingested": n_samples,
                "frauds_learned": n_fraud,
                "pre_update_log_loss": round(pre_loss, 5),
                "post_update_log_loss": round(post_loss, 5),
                "running_average_loss": round(self.stats.running_loss, 5),
                "cumulative_samples_seen": self.stats.total_samples_ingested,
                "cumulative_frauds_seen": self.stats.total_frauds_learned,
            }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns positive class (fraud) probabilities from online model."""
        with self._lock:
            if not self.is_initialized:
                raise RuntimeError("Online model is not initialized.")
            probas = self.model.predict_proba(X.values)
            return probas[:, 1]

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves current online model state to disk."""
        with self._lock:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"Saved online model snapshot to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "OnlineFraudLearner":
        """Loads online model instance from disk."""
        instance = joblib.load(filepath)
        if not isinstance(instance, OnlineFraudLearner):
            raise TypeError(f"Object at {filepath} is not an OnlineFraudLearner instance.")
        return instance
