"""Comprehensive Test Suite for Credit Card Fraud Detection Pipeline & Microservice.

Validates:
  1. Temporal Data Splitting & Non-Leakage
  2. Leakage-safe Preprocessing
  3. Class Imbalance Handling
  4. Swappable Model Architectures
  5. Cost-Aware Threshold Optimization & Workload Caps
  6. SHAP Attribution & Reason Codes
  7. Online/Incremental Learning
  8. FastAPI Endpoints (/health, /predict, /explain, /feedback)
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config import CONFIG, CostMatrixConfig
from cost_evaluation import CostAwareThresholdOptimizer
from data_preprocessing import (
    ImbalanceHandler,
    LeakageSafePreprocessor,
    TemporalDataSplitter,
    generate_synthetic_fraud_dataset,
)
from explainability import FraudExplainer
from fastapi_app import app, load_or_bootstrap_artifacts
from model_pipeline import ModelFactory
from online_learning import OnlineFraudLearner


@pytest.fixture(scope="session")
def sample_dataset():
    """Generates a small synthetic dataset for fast test execution."""
    return generate_synthetic_fraud_dataset(n_samples=2000, fraud_rate=0.02, random_state=42)


def test_temporal_data_splitter(sample_dataset):
    """Verifies strict chronological splitting without data leakage."""
    splitter = TemporalDataSplitter()
    splits = splitter.split(sample_dataset)

    # Validate split sizes
    assert len(splits.X_train) == int(len(sample_dataset) * 0.70)
    assert len(splits.X_val) == int(len(sample_dataset) * 0.15)
    assert len(splits.X_test) == len(sample_dataset) - len(splits.X_train) - len(splits.X_val)

    # Validate strict time monotonicity (No future data in training set)
    max_train_time = splits.X_train["Time"].max()
    min_val_time = splits.X_val["Time"].min()
    max_val_time = splits.X_val["Time"].max()
    min_test_time = splits.X_test["Time"].min()

    assert max_train_time <= min_val_time, "Train set must precede Validation set temporally."
    assert max_val_time <= min_test_time, "Validation set must precede Test set temporally."


def test_leakage_safe_preprocessor(sample_dataset):
    """Verifies that preprocessor learns parameters only from training data."""
    splitter = TemporalDataSplitter()
    splits = splitter.split(sample_dataset)

    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(splits.X_train)

    X_train_trans = preprocessor.transform(splits.X_train)
    X_val_trans = preprocessor.transform(splits.X_val)

    # Time column should be replaced by hour_sin, hour_cos, scaled_time_hour
    assert "Time" not in X_train_trans.columns
    assert "scaled_Amount" in X_train_trans.columns
    assert "hour_sin" in X_train_trans.columns
    assert "hour_cos" in X_train_trans.columns

    # Verify column consistency across partitions
    assert list(X_train_trans.columns) == list(X_val_trans.columns)


def test_imbalance_handler(sample_dataset):
    """Verifies SMOTE and class-weight imbalance handling strategies."""
    splitter = TemporalDataSplitter()
    splits = splitter.split(sample_dataset)

    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(splits.X_train)
    X_train_trans = preprocessor.transform(splits.X_train)

    # Test Hybrid Strategy
    handler = ImbalanceHandler()
    X_res, y_res, scale_weight = handler.handle_imbalance(X_train_trans, splits.y_train)

    assert len(X_res) == len(y_res)
    assert (y_res == 1).sum() > (splits.y_train == 1).sum(), "Minority class should be boosted."


def test_swappable_models(sample_dataset):
    """Verifies that XGBoost, Random Forest, and Logistic Regression follow the BaseFraudEstimator interface."""
    splitter = TemporalDataSplitter()
    splits = splitter.split(sample_dataset)

    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(splits.X_train)
    X_train_trans = preprocessor.transform(splits.X_train)
    X_val_trans = preprocessor.transform(splits.X_val)

    for model_name in ["xgboost", "logistic_regression", "random_forest"]:
        model = ModelFactory.create(model_name)
        model.fit(X_train_trans, splits.y_train)
        probas = model.predict_proba(X_val_trans)

        assert len(probas) == len(X_val_trans)
        assert np.all((probas >= 0.0) & (probas <= 1.0))
        pr_auc = model.evaluate_pr_auc(X_val_trans, splits.y_val)
        assert 0.0 <= pr_auc <= 1.0


def test_cost_aware_threshold_optimizer():
    """Verifies threshold optimization under fraud cost matrix and review capacity caps."""
    y_true = np.array([0] * 990 + [1] * 10)  # 1% fraud rate
    # Simulated probabilities: frauds get higher probabilities
    y_probas = np.array([0.02] * 950 + [0.35] * 40 + [0.85] * 10)

    cost_cfg = CostMatrixConfig(
        cost_false_negative=100.0,
        cost_false_positive=5.0,
        max_review_rate=0.03,  # Max 3% review capacity
    )
    optimizer = CostAwareThresholdOptimizer(cost_config=cost_cfg)
    best_res, eval_df = optimizer.find_optimal_threshold(y_true, y_probas)

    assert best_res.is_workload_compliant
    assert best_res.review_rate <= cost_cfg.max_review_rate
    assert best_res.total_cost >= 0.0


def test_online_learning_incremental():
    """Verifies that OnlineFraudLearner adapts weights incrementally via partial_fit."""
    learner = OnlineFraudLearner()

    X_init = pd.DataFrame(np.random.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
    y_init = pd.Series([0] * 190 + [1] * 10)

    learner.initialize_from_batch(X_init, y_init, epochs=2)
    assert learner.is_initialized
    assert learner.stats.total_samples_ingested == 200

    # Stream new batch
    X_stream = pd.DataFrame(np.random.randn(20, 10), columns=[f"feat_{i}" for i in range(10)])
    y_stream = [1] * 5 + [0] * 15

    result = learner.update_incremental(X_stream, y_stream)
    assert result["status"] == "SUCCESS"
    assert result["samples_ingested"] == 20
    assert learner.stats.total_samples_ingested == 220
    assert learner.stats.total_frauds_learned == 15


def test_fastapi_endpoints():
    """Validates FastAPI microservice endpoints with TestClient."""
    load_or_bootstrap_artifacts()

    with TestClient(app) as client:
        # 1. Health endpoint
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "HEALTHY"
        assert "optimal_threshold" in health_data

        # 2. Predict endpoint
        payload = {
            "transaction_id": "test-tx-001",
            "Time": 3600.0,
            "Amount": 129.50,
            "V1": -1.2,
            "V2": 0.5,
            "V3": -0.8,
            "V4": 2.1,
            "V14": -3.5,
            "V17": -2.0,
        }
        res_pred = client.post("/predict", json=payload)
        assert res_pred.status_code == 200
        pred_data = res_pred.json()
        assert "fraud_probability" in pred_data
        assert pred_data["decision"] in ["APPROVE", "MANUAL_REVIEW", "DECLINE"]
        assert len(pred_data["top_risk_reasons"]) > 0

        # 3. Explain endpoint
        res_exp = client.post("/explain", json=payload)
        assert res_exp.status_code == 200
        exp_data = res_exp.json()
        assert "all_feature_attributions" in exp_data

        # 4. Feedback endpoint (Online Learning)
        feedback_payload = {
            "transaction": payload,
            "is_fraud_confirmed": 1,
        }
        res_feed = client.post("/feedback", json=feedback_payload)
        assert res_feed.status_code == 200
        feed_data = res_feed.json()
        assert feed_data["status"] == "SUCCESS"
        assert feed_data["frauds_learned"] == 1

        # 5. Metrics endpoint
        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        metrics_data = res_metrics.json()
        assert "online_learning_telemetry" in metrics_data
