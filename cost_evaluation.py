"""Cost-Aware Threshold Optimization & Performance Evaluation Module.

Implements business-oriented decision threshold optimization based on a
custom fraud loss matrix (FN cost vs. FP investigation cost) subject to
explicit manual review workload budget constraints, alongside strict PR-AUC evaluation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from config import CONFIG, CostMatrixConfig

logger = logging.getLogger(__name__)


@dataclass
class ThresholdEvaluationResult:
    """Detailed performance and financial outcome metrics at a specific decision threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    review_rate: float
    is_workload_compliant: bool
    total_cost: float
    cost_per_transaction: float
    savings_vs_zero_defense: float


@dataclass
class OptimizationSummary:
    """Encapsulates the complete outcome of threshold search and validation."""

    optimal_threshold: float
    pr_auc: float
    roc_auc: float
    baseline_cost: float
    optimal_cost: float
    cost_reduction_pct: float
    review_rate: float
    precision_at_optimal: float
    recall_at_optimal: float
    max_review_rate_cap: float
    cost_matrix: Dict[str, float]


class CostAwareThresholdOptimizer:
    """Optimizes probability decision thresholds to minimize financial fraud loss

    under strict human operational capacity constraints.
    """

    def __init__(self, cost_config: CostMatrixConfig = CONFIG.cost):
        self.cost_config = cost_config

    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_probas: np.ndarray,
        threshold: float,
    ) -> ThresholdEvaluationResult:
        """Calculates confusion matrix, metrics, and financial loss for a given threshold."""
        y_pred = (y_probas >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_total = len(y_true)

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-9, (precision + recall))
        review_rate = (tp + fp) / max(1, n_total)

        total_cost = (
            fn * self.cost_config.cost_false_negative
            + fp * self.cost_config.cost_false_positive
            + tp * self.cost_config.cost_true_positive
            + tn * self.cost_config.cost_true_negative
        )
        cost_per_tx = total_cost / max(1, n_total)

        # Baseline cost if all frauds were missed (0 defense)
        total_fraud_count = tp + fn
        zero_defense_cost = total_fraud_count * self.cost_config.cost_false_negative
        savings = zero_defense_cost - total_cost

        is_compliant = review_rate <= self.cost_config.max_review_rate

        return ThresholdEvaluationResult(
            threshold=round(threshold, 4),
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            review_rate=float(review_rate),
            is_workload_compliant=bool(is_compliant),
            total_cost=float(total_cost),
            cost_per_transaction=float(cost_per_tx),
            savings_vs_zero_defense=float(savings),
        )

    def find_optimal_threshold(
        self,
        y_val_true: Union[pd.Series, np.ndarray],
        y_val_probas: Union[pd.Series, np.ndarray],
        threshold_steps: int = 200,
    ) -> Tuple[ThresholdEvaluationResult, pd.DataFrame]:
        """Performs grid search across probability thresholds to identify the cost-optimal decision point.

        Strictly enforces review rate <= max_review_rate. If no threshold strictly complies,
        it falls back to the lowest review rate threshold with minimal cost.

        Args:
            y_val_true: Ground truth binary labels for validation split.
            y_val_probas: Model predicted probabilities for positive class (fraud).
            threshold_steps: Number of candidate thresholds to evaluate.

        Returns:
            Tuple of (Optimal Threshold Result, DataFrame of all evaluated thresholds).
        """
        y_true_arr = np.asarray(y_val_true).astype(int)
        y_prob_arr = np.asarray(y_val_probas)

        # Combine uniform grid with exact precision-recall inflection points if positive class is present
        if len(np.unique(y_true_arr)) > 1 and np.sum(y_true_arr == 1) > 0:
            _, _, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
        else:
            pr_thresholds = np.array([])

        grid_thresholds = np.linspace(0.005, 0.995, threshold_steps)
        candidate_thresholds = np.unique(np.clip(np.concatenate([grid_thresholds, pr_thresholds]), 0.001, 0.999))
        candidate_thresholds = np.sort(candidate_thresholds)

        evaluations: List[ThresholdEvaluationResult] = []

        for thresh in candidate_thresholds:
            res = self.evaluate_threshold(y_true_arr, y_prob_arr, thresh)
            evaluations.append(res)

        eval_df = pd.DataFrame([asdict(e) for e in evaluations])

        # Filter by operational review budget constraint
        compliant_df = eval_df[eval_df["is_workload_compliant"]]

        if not compliant_df.empty:
            # Pick the threshold that minimizes total financial cost
            best_idx = compliant_df["total_cost"].idxmin()
            best_row = compliant_df.loc[best_idx]
            logger.info(
                f"Optimal threshold found within workload cap ({self.cost_config.max_review_rate:.2%}): "
                f"Threshold={best_row['threshold']:.4f}, Total Cost=${best_row['total_cost']:,.2f}, "
                f"Review Rate={best_row['review_rate']:.3%}"
            )
        else:
            logger.warning(
                f"No candidate threshold satisfied strict review cap of {self.cost_config.max_review_rate:.2%}. "
                "Selecting minimum cost threshold across all candidates."
            )
            best_idx = eval_df["total_cost"].idxmin()
            best_row = eval_df.loc[best_idx]

        optimal_result = ThresholdEvaluationResult(**best_row.to_dict())
        return optimal_result, eval_df

    def generate_summary(
        self,
        y_test_true: Union[pd.Series, np.ndarray],
        y_test_probas: Union[pd.Series, np.ndarray],
        optimal_threshold: float,
    ) -> OptimizationSummary:
        """Evaluates out-of-time test set using the tuned optimal threshold and PR-AUC."""
        y_test_arr = np.asarray(y_test_true).astype(int)
        y_prob_arr = np.asarray(y_test_probas)

        pr_auc = float(average_precision_score(y_test_arr, y_prob_arr))
        roc_auc = float(roc_auc_score(y_test_arr, y_prob_arr))

        # Default 0.50 threshold result vs tuned optimal threshold
        default_res = self.evaluate_threshold(y_test_arr, y_prob_arr, 0.50)
        optimal_res = self.evaluate_threshold(y_test_arr, y_prob_arr, optimal_threshold)

        cost_reduction = (
            (default_res.total_cost - optimal_res.total_cost) / max(1e-6, default_res.total_cost)
        ) * 100.0

        summary = OptimizationSummary(
            optimal_threshold=float(optimal_threshold),
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            baseline_cost=default_res.total_cost,
            optimal_cost=optimal_res.total_cost,
            cost_reduction_pct=float(cost_reduction),
            review_rate=optimal_res.review_rate,
            precision_at_optimal=optimal_res.precision,
            recall_at_optimal=optimal_res.recall,
            max_review_rate_cap=self.cost_config.max_review_rate,
            cost_matrix=asdict(self.cost_config),
        )

        logger.info(
            f"=== Final Evaluation Summary (Out-of-Time Test Set) ===\n"
            f"PR-AUC (Primary Metric): {pr_auc:.5f}\n"
            f"ROC-AUC:                 {roc_auc:.5f}\n"
            f"Decision Threshold:      {optimal_threshold:.4f}\n"
            f"Precision:               {optimal_res.precision:.4f}\n"
            f"Recall:                  {optimal_res.recall:.4f}\n"
            f"Review Rate:             {optimal_res.review_rate:.4%} (Cap: {self.cost_config.max_review_rate:.2%})\n"
            f"Total Cost at Optimal:   ${optimal_res.total_cost:,.2f} (vs Baseline: ${default_res.total_cost:,.2f})\n"
            f"Cost Reduction:          {cost_reduction:.2f}%\n"
            f"========================================================"
        )

        return summary

    def save_threshold_metadata(
        self, summary: OptimizationSummary, filepath: Union[str, Path]
    ) -> None:
        """Persists threshold metadata to a JSON artifact for the FastAPI service."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        logger.info(f"Saved threshold metadata to {path}")
