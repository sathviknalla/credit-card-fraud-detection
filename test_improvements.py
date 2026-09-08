"""Extended Test Suite for Phase 5-8 Improvements.

Tests for:
  - Phase 5: CalibrationWrapper (Platt / Isotonic Regression)
  - Phase 6: StackedFraudEnsemble (OOF meta-learner)
  - Phase 7: FraudCaseManager (state machine, audit trail, priority queue)
  - Phase 8: ModelRegistry + ChampionChallengerEvaluator + RetrainingScheduler
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    LeakageSafePreprocessor,
    TemporalDataSplitter,
    generate_synthetic_fraud_dataset,
)
from model_pipeline import ModelFactory


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dataset():
    """Shared small synthetic dataset for fast tests."""
    return generate_synthetic_fraud_dataset(n_samples=1500, fraud_rate=0.05, random_state=99)


@pytest.fixture(scope="module")
def prepared_splits(small_dataset):
    """Pre-split and pre-processed dataset for all phase tests."""
    splitter = TemporalDataSplitter()
    splits = splitter.split(small_dataset)

    preprocessor = LeakageSafePreprocessor()
    preprocessor.fit(splits.X_train)

    X_train = preprocessor.transform(splits.X_train)
    X_val = preprocessor.transform(splits.X_val)
    X_test = preprocessor.transform(splits.X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": splits.y_train.reset_index(drop=True),
        "y_val": splits.y_val.reset_index(drop=True),
        "y_test": splits.y_test.reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def fitted_base_model(prepared_splits):
    """Returns a fitted XGBoost base model for calibration/ensemble tests."""
    model = ModelFactory.create("xgboost")
    model.fit(prepared_splits["X_train"], prepared_splits["y_train"])
    return model


# ---------------------------------------------------------------------------
# Phase 5: Probability Calibration Tests
# ---------------------------------------------------------------------------

class TestCalibrationWrapper:
    def test_isotonic_calibration_returns_probabilities(self, fitted_base_model, prepared_splits):
        """Calibrated model should return valid probabilities in [0,1]."""
        from model_calibration import CalibrationWrapper

        cal = CalibrationWrapper(base_model=fitted_base_model, method="isotonic")
        report = cal.fit_calibration(prepared_splits["X_val"], prepared_splits["y_val"])

        probas = cal.predict_proba(prepared_splits["X_test"])
        assert len(probas) == len(prepared_splits["X_test"])
        assert np.all((probas >= 0.0) & (probas <= 1.0)), "All probabilities must be in [0, 1]"

    def test_sigmoid_calibration_produces_report(self, fitted_base_model, prepared_splits):
        """Platt Scaling should produce a CalibrationReport with valid Brier scores."""
        from model_calibration import CalibrationWrapper, CalibrationReport

        cal = CalibrationWrapper(base_model=fitted_base_model, method="sigmoid")
        report = cal.fit_calibration(prepared_splits["X_val"], prepared_splits["y_val"])

        assert isinstance(report, CalibrationReport)
        assert report.brier_score_before > 0.0
        assert report.brier_score_after > 0.0
        assert report.n_calibration_samples == len(prepared_splits["X_val"])

    def test_invalid_method_raises(self, fitted_base_model):
        """Invalid calibration method should raise ValueError."""
        from model_calibration import CalibrationWrapper

        with pytest.raises(ValueError, match="method must be one of"):
            CalibrationWrapper(base_model=fitted_base_model, method="invalid_method")

    def test_fallback_before_calibration(self, fitted_base_model, prepared_splits):
        """predict_proba before fit_calibration should fall back to raw model (with warning)."""
        from model_calibration import CalibrationWrapper

        cal = CalibrationWrapper(base_model=fitted_base_model, method="isotonic")
        # No fit_calibration call
        probas = cal.predict_proba(prepared_splits["X_test"])
        assert len(probas) == len(prepared_splits["X_test"])

    def test_predict_binary_thresholding(self, fitted_base_model, prepared_splits):
        """predict() should return binary 0/1 values only."""
        from model_calibration import CalibrationWrapper

        cal = CalibrationWrapper(base_model=fitted_base_model, method="isotonic")
        cal.fit_calibration(prepared_splits["X_val"], prepared_splits["y_val"])
        preds = cal.predict(prepared_splits["X_test"], threshold=0.5)
        assert set(preds).issubset({0, 1})

    def test_save_and_load_roundtrip(self, fitted_base_model, prepared_splits):
        """CalibrationWrapper should survive a save/load cycle."""
        from model_calibration import CalibrationWrapper

        cal = CalibrationWrapper(base_model=fitted_base_model, method="isotonic")
        cal.fit_calibration(prepared_splits["X_val"], prepared_splits["y_val"])
        probas_before = cal.predict_proba(prepared_splits["X_test"])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            cal.save(f.name)
            loaded = CalibrationWrapper.load(f.name)

        probas_after = loaded.predict_proba(prepared_splits["X_test"])
        np.testing.assert_allclose(probas_before, probas_after, rtol=1e-5)

    def test_calibration_curve_plot_returns_figure(self, fitted_base_model, prepared_splits):
        """plot_calibration_curve should return a matplotlib Figure without errors."""
        from model_calibration import CalibrationWrapper
        import matplotlib.pyplot as plt

        cal = CalibrationWrapper(base_model=fitted_base_model, method="sigmoid")
        cal.fit_calibration(prepared_splits["X_val"], prepared_splits["y_val"])

        fig = cal.plot_calibration_curve(
            prepared_splits["X_test"],
            prepared_splits["y_test"],
            n_bins=5,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 6: Stacked Ensemble Tests
# ---------------------------------------------------------------------------

class TestStackedFraudEnsemble:
    def test_ensemble_returns_valid_probabilities(self, prepared_splits):
        """StackedFraudEnsemble.predict_proba should return values in [0,1]."""
        from ensemble_pipeline import StackedFraudEnsemble

        ensemble = StackedFraudEnsemble(
            base_model_names=["xgboost", "logistic_regression"],
            n_folds=2,
        )
        ensemble.fit(
            prepared_splits["X_train"],
            prepared_splits["y_train"],
            prepared_splits["X_val"],
            prepared_splits["y_val"],
        )
        probas = ensemble.predict_proba(prepared_splits["X_test"])
        assert len(probas) == len(prepared_splits["X_test"])
        assert np.all((probas >= 0.0) & (probas <= 1.0))

    def test_ensemble_produces_report(self, prepared_splits):
        """EnsembleReport should be generated after fitting with a validation set."""
        from ensemble_pipeline import StackedFraudEnsemble, EnsembleReport

        ensemble = StackedFraudEnsemble(
            base_model_names=["xgboost", "logistic_regression"],
            n_folds=2,
        )
        ensemble.fit(
            prepared_splits["X_train"],
            prepared_splits["y_train"],
            prepared_splits["X_val"],
            prepared_splits["y_val"],
        )
        assert ensemble.report is not None
        assert isinstance(ensemble.report, EnsembleReport)
        assert 0.0 <= ensemble.report.ensemble_pr_auc <= 1.0

    def test_meta_learner_coefficients_sum_to_meaningful_weights(self, prepared_splits):
        """Meta-learner should assign non-trivial coefficients to each base model."""
        from ensemble_pipeline import StackedFraudEnsemble

        ensemble = StackedFraudEnsemble(
            base_model_names=["xgboost", "logistic_regression"],
            n_folds=2,
        )
        ensemble.fit(prepared_splits["X_train"], prepared_splits["y_train"])
        assert ensemble.meta_learner is not None
        coef_sum = np.sum(np.abs(ensemble.meta_learner.coef_))
        assert coef_sum > 0.0, "Meta-learner should have non-zero coefficients."

    def test_ensemble_predict_returns_binary(self, prepared_splits):
        """predict() should return only 0s and 1s."""
        from ensemble_pipeline import StackedFraudEnsemble

        ensemble = StackedFraudEnsemble(
            base_model_names=["logistic_regression"],
            n_folds=2,
        )
        ensemble.fit(prepared_splits["X_train"], prepared_splits["y_train"])
        preds = ensemble.predict(prepared_splits["X_test"], threshold=0.5)
        assert set(preds).issubset({0, 1})

    def test_ensemble_save_load_roundtrip(self, prepared_splits):
        """StackedFraudEnsemble should survive a save/load cycle."""
        from ensemble_pipeline import StackedFraudEnsemble

        ensemble = StackedFraudEnsemble(
            base_model_names=["logistic_regression"],
            n_folds=2,
        )
        ensemble.fit(prepared_splits["X_train"], prepared_splits["y_train"])
        probas_before = ensemble.predict_proba(prepared_splits["X_test"])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            ensemble.save(f.name)
            loaded = StackedFraudEnsemble.load(f.name)

        probas_after = loaded.predict_proba(prepared_splits["X_test"])
        np.testing.assert_allclose(probas_before, probas_after, rtol=1e-5)

    def test_unfitted_ensemble_raises(self, prepared_splits):
        """predict_proba before fit should raise RuntimeError."""
        from ensemble_pipeline import StackedFraudEnsemble

        ensemble = StackedFraudEnsemble(n_folds=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            ensemble.predict_proba(prepared_splits["X_test"])


# ---------------------------------------------------------------------------
# Phase 7: Fraud Case Management Tests
# ---------------------------------------------------------------------------

class TestFraudCaseManager:
    def _make_manager(self):
        from case_manager import FraudCaseManager
        return FraudCaseManager()

    def test_open_case_creates_case_with_correct_priority(self):
        """High probability transaction should receive CRITICAL priority."""
        manager = self._make_manager()
        case = manager.open_case(
            transaction_id="tx-001",
            fraud_probability=0.95,
            expected_loss=600.0,
            transaction_amount=600.0,
        )
        from case_manager import CasePriority, CaseStatus
        assert case.priority == CasePriority.CRITICAL
        assert case.status == CaseStatus.FLAGGED
        assert len(case.audit_trail) == 1
        assert case.audit_trail[0].action == "CASE_CREATED"

    def test_state_transition_flagged_to_under_review(self):
        """Valid transition FLAGGED → UNDER_REVIEW should succeed."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        case = manager.open_case("tx-002", 0.80, 300.0, 300.0)
        updated = manager.transition_status(
            case.case_id, CaseStatus.UNDER_REVIEW, analyst_id="analyst-1", notes="Opening case"
        )
        assert updated.status == CaseStatus.UNDER_REVIEW
        assert updated.assigned_analyst == "analyst-1"
        assert len(updated.audit_trail) == 2

    def test_invalid_transition_raises_value_error(self):
        """Skipping UNDER_REVIEW to go directly FLAGGED → CONFIRMED should raise."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        case = manager.open_case("tx-003", 0.70, 100.0, 100.0)
        with pytest.raises(ValueError, match="Invalid transition"):
            manager.transition_status(case.case_id, CaseStatus.CONFIRMED_FRAUD)

    def test_full_lifecycle_confirmed_fraud(self):
        """A case should be able to transition through the full fraud lifecycle."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        case = manager.open_case("tx-004", 0.92, 750.0, 750.0)

        manager.transition_status(case.case_id, CaseStatus.UNDER_REVIEW, "analyst-1")
        manager.transition_status(case.case_id, CaseStatus.CONFIRMED_FRAUD, "analyst-1", "Confirmed via bank records")

        final_case = manager.get_case(case.case_id)
        assert final_case.status == CaseStatus.CONFIRMED_FRAUD
        assert len(final_case.audit_trail) == 3

    def test_terminal_state_blocks_further_transitions(self):
        """No transitions should be allowed from terminal states (CONFIRMED_FRAUD)."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        case = manager.open_case("tx-005", 0.91, 400.0, 400.0)
        manager.transition_status(case.case_id, CaseStatus.UNDER_REVIEW)
        manager.transition_status(case.case_id, CaseStatus.CONFIRMED_FRAUD, notes="Confirmed")
        with pytest.raises(ValueError, match="Invalid transition"):
            manager.transition_status(case.case_id, CaseStatus.CLEARED)

    def test_analyst_queue_sorted_by_priority(self):
        """Analyst queue should return CRITICAL cases first."""
        from case_manager import FraudCaseManager, CasePriority
        manager = self._make_manager()

        manager.open_case("tx-low", 0.20, 10.0, 50.0)
        manager.open_case("tx-critical", 0.95, 800.0, 800.0)
        manager.open_case("tx-medium", 0.55, 80.0, 100.0)

        queue = manager.get_analyst_queue()
        assert queue[0].priority == CasePriority.CRITICAL

    def test_statistics_returns_correct_counts(self):
        """Statistics should reflect correct case counts per status."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        manager.open_case("tx-a", 0.80, 200.0, 200.0)
        case_b = manager.open_case("tx-b", 0.60, 100.0, 100.0)
        manager.transition_status(case_b.case_id, CaseStatus.UNDER_REVIEW)
        manager.transition_status(case_b.case_id, CaseStatus.CLEARED, notes="Legit")

        stats = manager.get_statistics()
        assert stats["total_cases"] == 2
        assert stats["by_status"]["FLAGGED"] == 1
        assert stats["by_status"]["CLEARED"] == 1
        assert stats["resolved_count"] == 1

    def test_audit_trail_is_complete(self):
        """get_full_audit_trail should return all transition events."""
        from case_manager import FraudCaseManager, CaseStatus
        manager = self._make_manager()
        case = manager.open_case("tx-audit", 0.88, 350.0, 350.0)
        manager.transition_status(case.case_id, CaseStatus.UNDER_REVIEW, "analyst-A", "Started")
        manager.transition_status(case.case_id, CaseStatus.ESCALATED, "analyst-A", "Needs senior review")
        manager.transition_status(case.case_id, CaseStatus.CONFIRMED_FRAUD, "senior-B", "Escalation confirmed")

        trail = manager.get_full_audit_trail(case.case_id)
        assert len(trail) == 4  # CREATE + 3 transitions
        assert trail[0]["action"] == "CASE_CREATED"
        assert trail[-1]["to_status"] == "CONFIRMED_FRAUD"

    def test_unknown_case_raises_key_error(self):
        """Getting a non-existent case should raise KeyError."""
        from case_manager import FraudCaseManager
        manager = self._make_manager()
        with pytest.raises(KeyError):
            manager.get_case("non-existent-id")


# ---------------------------------------------------------------------------
# Phase 8: Model Registry & Champion-Challenger Tests
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def _make_registry(self, tmp_path):
        from retraining_scheduler import ModelRegistry
        return ModelRegistry(registry_path=tmp_path / "test_registry.json")

    def _make_version(self, version_id, pr_auc_val, is_champion=False):
        from retraining_scheduler import ModelVersion
        return ModelVersion(
            version_id=version_id,
            model_type="xgboost",
            training_timestamp="2026-01-01T00:00:00Z",
            pr_auc_val=pr_auc_val,
            pr_auc_test=pr_auc_val - 0.01,
            optimal_threshold=0.45,
            imbalance_strategy="hybrid",
            n_training_samples=10000,
            trigger_reason="MANUAL",
            artifact_path="/tmp/model.joblib",
            is_champion=is_champion,
        )

    def test_register_and_retrieve_champion(self, tmp_path):
        """Registering a champion version should allow retrieval."""
        registry = self._make_registry(tmp_path)
        v1 = self._make_version("v1", pr_auc_val=0.85, is_champion=True)
        registry.register_version(v1)

        champion = registry.get_champion()
        assert champion is not None
        assert champion.version_id == "v1"
        assert champion.is_champion is True

    def test_only_one_champion_at_a_time(self, tmp_path):
        """Registering a second champion should demote the first."""
        registry = self._make_registry(tmp_path)
        v1 = self._make_version("v1", pr_auc_val=0.80, is_champion=True)
        v2 = self._make_version("v2", pr_auc_val=0.85, is_champion=True)
        registry.register_version(v1)
        registry.register_version(v2)

        champions = [v for v in registry.list_versions() if v["is_champion"]]
        assert len(champions) == 1
        assert champions[0]["version_id"] == "v2"

    def test_set_champion_promotes_correct_version(self, tmp_path):
        """set_champion should promote v1 and demote v2."""
        registry = self._make_registry(tmp_path)
        v1 = self._make_version("v1", 0.80, is_champion=False)
        v2 = self._make_version("v2", 0.85, is_champion=True)
        registry.register_version(v1)
        registry.register_version(v2)

        registry.set_champion("v1")
        champion = registry.get_champion()
        assert champion.version_id == "v1"

    def test_rollback_to_previous_version(self, tmp_path):
        """rollback_to should revert champion to an older version."""
        registry = self._make_registry(tmp_path)
        v1 = self._make_version("v1", 0.80, is_champion=True)
        v2 = self._make_version("v2", 0.85, is_champion=True)
        registry.register_version(v1)
        registry.register_version(v2)

        registry.rollback_to("v1")
        assert registry.get_champion().version_id == "v1"

    def test_get_best_version_by_pr_auc(self, tmp_path):
        """get_best_version should return version with highest pr_auc_val."""
        registry = self._make_registry(tmp_path)
        for i, score in enumerate([0.70, 0.85, 0.78]):
            registry.register_version(self._make_version(f"v{i}", score))

        best = registry.get_best_version(metric="pr_auc_val")
        assert best is not None
        assert best.version_id == "v1"

    def test_registry_persists_and_reloads(self, tmp_path):
        """Registry should persist to disk and reload with full data."""
        from retraining_scheduler import ModelRegistry
        path = tmp_path / "reg.json"

        reg1 = ModelRegistry(registry_path=path)
        reg1.register_version(self._make_version("v-persist", 0.88, is_champion=True))

        reg2 = ModelRegistry(registry_path=path)
        versions = reg2.list_versions()
        assert len(versions) == 1
        assert versions[0]["version_id"] == "v-persist"
        assert versions[0]["is_champion"] is True

    def test_list_versions_is_empty_initially(self, tmp_path):
        """A fresh registry should have no versions."""
        registry = self._make_registry(tmp_path)
        assert registry.list_versions() == []


class TestChampionChallengerEvaluator:
    def test_challenger_wins_when_better(self):
        """Challenger with significantly higher PR-AUC should be promoted."""
        from retraining_scheduler import ChampionChallengerEvaluator

        evaluator = ChampionChallengerEvaluator(min_improvement_threshold=0.005)
        y_true = np.array([0] * 900 + [1] * 100)

        # Champion: moderate predictions (imperfect separation)
        champion_probas = np.array([0.1] * 850 + [0.7] * 50 + [0.4] * 100)
        # Challenger: better separation
        challenger_probas = np.array([0.05] * 850 + [0.2] * 50 + [0.9] * 100)

        result = evaluator.evaluate(
            champion_probas, challenger_probas, y_true, "champ-001", "chal-002"
        )
        assert result.winner == "challenger"
        assert result.promote_challenger is True

    def test_champion_wins_when_challenger_is_worse(self):
        """Champion should win when challenger performs comparably or worse."""
        from retraining_scheduler import ChampionChallengerEvaluator

        evaluator = ChampionChallengerEvaluator(min_improvement_threshold=0.005)
        y_true = np.array([0] * 900 + [1] * 100)

        # Champion: good predictions
        champion_probas = np.array([0.05] * 850 + [0.2] * 50 + [0.9] * 100)
        # Challenger: worse predictions (imperfect separation)
        challenger_probas = np.array([0.1] * 850 + [0.7] * 50 + [0.4] * 100)

        result = evaluator.evaluate(
            champion_probas, challenger_probas, y_true, "champ-001", "chal-002"
        )
        assert result.winner == "champion"
        assert result.promote_challenger is False

    def test_delta_pr_auc_is_correctly_computed(self):
        """delta_pr_auc should be challenger_pr_auc - champion_pr_auc."""
        from retraining_scheduler import ChampionChallengerEvaluator
        from sklearn.metrics import average_precision_score

        evaluator = ChampionChallengerEvaluator()
        y_true = np.array([0] * 90 + [1] * 10)
        champion_probas = np.linspace(0, 0.5, 100)
        challenger_probas = np.linspace(0, 0.8, 100)

        result = evaluator.evaluate(champion_probas, challenger_probas, y_true, "c1", "c2")
        expected_delta = (
            average_precision_score(y_true, challenger_probas)
            - average_precision_score(y_true, champion_probas)
        )
        assert abs(result.delta_pr_auc - expected_delta) < 1e-5
