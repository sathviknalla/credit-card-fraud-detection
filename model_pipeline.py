"""Modular Model Pipeline and Swappable Estimator Architecture.

Implements an Object-Oriented design with an abstract base estimator interface,
concrete model implementations (XGBoost, Random Forest, Logistic Regression, LightGBM),
and a Model Factory pattern allowing seamless swapping of model backends.
"""

import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import xgboost as xgb

from config import CONFIG

logger = logging.getLogger(__name__)


class BaseFraudEstimator(abc.ABC):
    """Abstract Base Class for all fraud detection model estimators.

    Enforces a uniform interface for fitting, probability prediction,
    serialization, and hyperparameter introspection across different algorithms.
    """

    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.params = params or {}
        self.estimator: Any = None
        self.is_fitted: bool = False
        self.feature_names_: Optional[list] = None

    @abc.abstractmethod
    def build_estimator(self) -> Any:
        """Constructs and returns the underlying ML estimator instance."""
        pass

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        scale_pos_weight: Optional[float] = None,
    ) -> "BaseFraudEstimator":
        """Fits the underlying model with optional validation monitoring."""
        if self.estimator is None:
            if scale_pos_weight and "scale_pos_weight" in self.params:
                self.params["scale_pos_weight"] = scale_pos_weight
            self.estimator = self.build_estimator()

        self.feature_names_ = list(X_train.columns)
        logger.info(f"Training {self.model_name} with {len(X_train):,} samples and {X_train.shape[1]} features...")
        self._fit_internal(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        return self

    @abc.abstractmethod
    def _fit_internal(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Concrete algorithm-specific fitting routine."""
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated fraud probabilities for Class 1 (Fraud)."""
        if not self.is_fitted or self.estimator is None:
            raise RuntimeError(f"Model {self.model_name} must be fitted before predicting.")
        probas = self.estimator.predict_proba(X)
        # Return probability of positive class (fraud)
        return probas[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Applies a custom decision threshold to predict binary fraud classes (0/1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate_pr_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Computes Precision-Recall AUC (Average Precision Score)."""
        probas = self.predict_proba(X)
        score = float(average_precision_score(y, probas))
        logger.info(f"[{self.model_name}] PR-AUC (Average Precision): {score:.5f}")
        return score

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the fitted model wrapper to disk."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Successfully saved {self.model_name} to {target_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseFraudEstimator":
        """Deserializes model wrapper from disk."""
        model = joblib.load(filepath)
        if not isinstance(model, BaseFraudEstimator):
            raise TypeError(f"Loaded object from {filepath} is not an instance of BaseFraudEstimator.")
        return model


class XGBoostFraudModel(BaseFraudEstimator):
    """XGBoost Gradient Boosted Trees estimator optimized for imbalanced classification."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = CONFIG.models.xgboost_params.copy()
        if params:
            merged_params.update(params)
        super().__init__(model_name="XGBoost", params=merged_params)

    def build_estimator(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(**self.params)

    def _fit_internal(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        if X_val is not None and y_val is not None:
            self.estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.estimator.fit(X_train, y_train, verbose=False)


class RandomForestFraudModel(BaseFraudEstimator):
    """Random Forest Classifier with balanced class subsampling."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = CONFIG.models.random_forest_params.copy()
        if params:
            merged_params.update(params)
        super().__init__(model_name="RandomForest", params=merged_params)

    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self.params)

    def _fit_internal(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self.estimator.fit(X_train, y_train)


class LogisticRegressionFraudModel(BaseFraudEstimator):
    """L2-regularized Logistic Regression baseline estimator."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = CONFIG.models.logistic_regression_params.copy()
        if params:
            merged_params.update(params)
        super().__init__(model_name="LogisticRegression", params=merged_params)

    def build_estimator(self) -> LogisticRegression:
        return LogisticRegression(**self.params)

    def _fit_internal(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self.estimator.fit(X_train, y_train)


class ModelFactory:
    """Factory for creating swappable fraud detection model instances."""

    _REGISTRY: Dict[str, type] = {
        "xgboost": XGBoostFraudModel,
        "random_forest": RandomForestFraudModel,
        "rf": RandomForestFraudModel,
        "logistic_regression": LogisticRegressionFraudModel,
        "lr": LogisticRegressionFraudModel,
    }

    @classmethod
    def register_model(cls, name: str, model_cls: type) -> None:
        """Allows dynamic registration of new custom model architectures."""
        cls._REGISTRY[name.lower()] = model_cls
        logger.info(f"Registered new model type '{name}' to ModelFactory.")

    @classmethod
    def create(cls, name: str, params: Optional[Dict[str, Any]] = None) -> BaseFraudEstimator:
        """Instantiates a registered fraud detection model by name.

        Args:
            name: Identifier string (e.g. 'xgboost', 'random_forest', 'logistic_regression').
            params: Optional dictionary of hyperparameter overrides.

        Returns:
            Fitted/unfitted instance conforming to BaseFraudEstimator interface.
        """
        normalized_name = name.lower()
        if normalized_name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(f"Unknown model name '{name}'. Available models: {available}")

        model_class = cls._REGISTRY[normalized_name]
        return model_class(params=params)
