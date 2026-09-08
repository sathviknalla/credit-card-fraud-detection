"""Global Configuration Module for Credit Card Fraud Detection System.

Defines all pipeline constants, cost parameters, model configurations,
and artifact storage locations in a centralized, type-safe manner.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CostMatrixConfig:
    """Financial parameters for cost-aware decision threshold optimization.

    Attributes:
        cost_false_negative: Average financial loss incurred when a fraudulent
            transaction goes undetected (chargeback fee + merchandise loss).
        cost_false_positive: Operational cost incurred when an automated alert
            requires manual review by a fraud analyst.
        cost_true_positive: Verification or step-up authentication cost when fraud
            is successfully intercepted.
        cost_true_negative: Nominal friction cost for a legitimate approved transaction.
        max_review_rate: Hard operational constraint representing the maximum fraction
            of total transactions that the human review team can audit (e.g., 1.5%).
    """

    cost_false_negative: float = 120.0
    cost_false_positive: float = 5.0
    cost_true_positive: float = 0.5
    cost_true_negative: float = 0.0
    max_review_rate: float = 0.015


@dataclass(frozen=True)
class DataConfig:
    """Dataset ingestion and temporal partitioning settings.

    Attributes:
        dataset_path: Path to the raw creditcard.csv dataset.
        time_column: Name of the transaction timestamp/offset column.
        target_column: Name of the binary fraud label column (1=Fraud, 0=Legitimate).
        amount_column: Name of the transaction amount column.
        pca_features: List of anonymized V1..V28 feature names.
        train_ratio: Fraction of earliest transactions used for model training.
        val_ratio: Fraction of middle transactions used for threshold tuning.
        test_ratio: Fraction of latest transactions used for final out-of-time test.
        random_state: Seed for reproducibility in sampling/shuffling.
    """

    dataset_path: Path = Path("data/creditcard.csv")
    time_column: str = "Time"
    target_column: str = "Class"
    amount_column: str = "Amount"
    pca_features: List[str] = field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class ImbalanceConfig:
    """Resampling and class weighting strategies for extreme imbalance handling.

    Attributes:
        strategy: Method to handle imbalance: 'smote', 'class_weight', 'hybrid', or 'none'.
        smote_sampling_strategy: Target minority-to-majority ratio for SMOTE (e.g. 0.05).
        smote_k_neighbors: Number of nearest neighbors for SMOTE synthetic sample creation.
        undersample_ratio: Ratio for optional RandomUnderSampler in hybrid mode.
    """

    strategy: str = "hybrid"  # options: 'smote', 'class_weight', 'hybrid', 'none'
    smote_sampling_strategy: float = 0.05
    smote_k_neighbors: int = 5
    undersample_ratio: Optional[float] = 0.20


@dataclass(frozen=True)
class ModelHyperparams:
    """Default hyperparameter grids and settings for swappable estimators."""

    xgboost_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,  # Dynamically computed if class_weight strategy used
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    random_forest_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 150,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    logistic_regression_params: Dict[str, object] = field(
        default_factory=lambda: {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
        }
    )

    online_sgd_params: Dict[str, object] = field(
        default_factory=lambda: {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "learning_rate": "optimal",
            "random_state": 42,
        }
    )


@dataclass
class PipelineConfig:
    """Master configuration encapsulating all subsystem settings."""

    data: DataConfig = field(default_factory=DataConfig)
    cost: CostMatrixConfig = field(default_factory=CostMatrixConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    models: ModelHyperparams = field(default_factory=ModelHyperparams)

    # Artifact Storage
    artifact_dir: Path = Path("artifacts")
    model_artifact_name: str = "fraud_model.joblib"
    preprocessor_artifact_name: str = "preprocessor.joblib"
    threshold_artifact_name: str = "threshold_metadata.json"
    online_model_artifact_name: str = "online_fraud_model.joblib"

    def __post_init__(self):
        """Ensure artifact directories exist."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)


# Default global singleton configuration instance
CONFIG = PipelineConfig()
