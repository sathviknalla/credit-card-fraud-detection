"""End-to-End Model Training, Threshold Optimization, and Serialization Pipeline.

Executes the complete workflow:
  1. Data Ingestion / Synthetic Generation
  2. Leakage-safe Chronological Splitting
  3. Feature Engineering & Scaling
  4. Train-set-only Imbalance Handling (SMOTE / Class Weights)
  5. Core Estimator Training & PR-AUC Evaluation
  6. Cost-Aware Threshold Optimization with Review Capacity Constraints
  7. Final Out-of-Time Test Set Evaluation
  8. Online Learning Baseline Initialization
  9. Artifact Serialization for Production Deployment
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from config import CONFIG, PipelineConfig
from cost_evaluation import CostAwareThresholdOptimizer
from data_preprocessing import (
    ImbalanceHandler,
    LeakageSafePreprocessor,
    TemporalDataSplitter,
    generate_synthetic_fraud_dataset,
)
from explainability import FraudExplainer
from model_pipeline import BaseFraudEstimator, ModelFactory
import ensemble_pipeline  # Registers the ensemble with ModelFactory
from online_learning import OnlineFraudLearner

logger = logging.getLogger("TrainPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_training_pipeline(
    model_type: str = "xgboost",
    imbalance_strategy: str = "hybrid",
    dataset_path: Path = CONFIG.data.dataset_path,
    artifact_dir: Path = CONFIG.artifact_dir,
    n_samples: Optional[int] = None,
) -> BaseFraudEstimator:
    """Executes the full model training and evaluation lifecycle."""
    logger.info("=== STARTING CREDIT CARD FRAUD DETECTION TRAINING PIPELINE ===")

    # 1. Ingest or generate dataset
    if dataset_path.exists():
        logger.info(f"Loading transaction dataset from {dataset_path}...")
        raw_df = pd.read_csv(dataset_path)
    else:
        sample_count = n_samples or 284807
        logger.warning(
            f"Dataset not found at {dataset_path}. Generating synthetic dataset with {sample_count:,} transactions..."
        )
        raw_df = generate_synthetic_fraud_dataset(n_samples=sample_count, fraud_rate=0.00172)

    # 2. Strict chronological split (70% Train / 15% Val / 15% Test)
    splitter = TemporalDataSplitter(config=CONFIG.data)
    splits = splitter.split(raw_df)

    # 3. Leakage-safe feature preprocessing (Fit on Train ONLY)
    logger.info("Fitting LeakageSafePreprocessor on training partition...")
    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(splits.X_train)

    X_train_trans = preprocessor.transform(splits.X_train)
    X_val_trans = preprocessor.transform(splits.X_val)
    X_test_trans = preprocessor.transform(splits.X_test)

    # 4. Handle class imbalance (Applied strictly to Train split)
    CONFIG.imbalance = CONFIG.imbalance.__class__(strategy=imbalance_strategy)
    imbalance_handler = ImbalanceHandler(config=CONFIG.imbalance)
    X_train_resampled, y_train_resampled, scale_pos_weight = imbalance_handler.handle_imbalance(
        X_train_trans, splits.y_train
    )

    # 5. Initialize & Train Core Model
    logger.info(f"Instantiating model architecture: '{model_type}' via ModelFactory...")
    model: BaseFraudEstimator = ModelFactory.create(name=model_type)

    model.fit(
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_val=X_val_trans,
        y_val=splits.y_val,
        scale_pos_weight=scale_pos_weight,
    )

    # 6. PR-AUC Evaluation on Validation Set
    val_pr_auc = model.evaluate_pr_auc(X_val_trans, splits.y_val)
    val_probas = model.predict_proba(X_val_trans)

    # 7. Cost-Aware Threshold Optimization on Validation Set
    logger.info("Optimizing decision threshold against financial fraud cost matrix...")
    optimizer = CostAwareThresholdOptimizer(cost_config=CONFIG.cost)
    optimal_thresh_res, _ = optimizer.find_optimal_threshold(
        y_val_true=splits.y_val,
        y_val_probas=val_probas,
    )
    optimal_threshold = optimal_thresh_res.threshold

    # 8. Unbiased Evaluation on Out-of-Time Test Set
    test_probas = model.predict_proba(X_test_trans)
    summary = optimizer.generate_summary(
        y_test_true=splits.y_test,
        y_test_probas=test_probas,
        optimal_threshold=optimal_threshold,
    )

    # 9. Initialize & Seed Online Learner Baseline
    logger.info("Initializing Online/Incremental Learning Engine...")
    online_learner = OnlineFraudLearner()
    # Seed with sample from training set
    sample_size = min(50000, len(X_train_trans))
    online_learner.initialize_from_batch(
        X_init=X_train_trans.iloc[:sample_size],
        y_init=splits.y_train.iloc[:sample_size],
        epochs=3,
    )

    # 10. Serialize All Production Artifacts
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / CONFIG.model_artifact_name
    preprocessor_path = artifact_dir / CONFIG.preprocessor_artifact_name
    threshold_path = artifact_dir / CONFIG.threshold_artifact_name
    online_model_path = artifact_dir / CONFIG.online_model_artifact_name

    model.save(model_path)
    import joblib
    joblib.dump(preprocessor, preprocessor_path)
    optimizer.save_threshold_metadata(summary, threshold_path)
    online_learner.save(online_model_path)

    logger.info(f"=== ALL PRODUCTION ARTIFACTS SERIALIZED TO: {artifact_dir.resolve()} ===")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic_regression", "ensemble"],
        help="Core estimator type to train",
    )
    parser.add_argument(
        "--imbalance",
        type=str,
        default="hybrid",
        choices=["smote", "class_weight", "hybrid", "none"],
        help="Imbalance handling strategy",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of synthetic samples to generate if dataset not found on disk",
    )
    args = parser.parse_args()

    run_training_pipeline(model_type=args.model, imbalance_strategy=args.imbalance, n_samples=args.samples)
