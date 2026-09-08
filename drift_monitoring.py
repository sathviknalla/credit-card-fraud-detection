"""Data & Concept Drift Monitoring Engine for Production Fraud Systems.

Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to detect distribution shifts in transaction feature streams and alert on model degradation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftReport:
    """Drift evaluation report for a single feature."""

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_status: str  # 'STABLE', 'MODERATE_SHIFT', 'SIGNIFICANT_DRIFT'
    is_drifted: bool


@dataclass
class SystemDriftSummary:
    """Comprehensive drift scorecard across all monitored features."""

    overall_system_status: str  # 'HEALTHY', 'WARNING', 'ACTION_REQUIRED'
    drifted_feature_count: int
    total_features_monitored: int
    max_psi_feature: str
    max_psi_value: float
    recommendation: str
    feature_reports: List[FeatureDriftReport]


class ConceptDriftMonitor:
    """Monitors streaming transaction features against baseline reference distributions."""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        buffer_size: int = 1000,
        psi_threshold_warning: float = 0.10,
        psi_threshold_drift: float = 0.25,
        ks_alpha: float = 0.05,
    ):
        self.reference_data = reference_data
        self.buffer_size = buffer_size
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_drift = psi_threshold_drift
        self.ks_alpha = ks_alpha

        self.stream_buffer: List[Dict[str, float]] = []
        self.monitored_columns: List[str] = []

        if reference_data is not None:
            self.set_reference_data(reference_data)

    def set_reference_data(self, df: pd.DataFrame) -> None:
        """Sets historical baseline distribution (e.g., training partition)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.monitored_columns = [col for col in numeric_cols if col not in ["Class", "Time"]]
        self.reference_data = df[self.monitored_columns].copy()
        logger.info(
            f"DriftMonitor initialized with reference data: {len(self.reference_data):,} samples, "
            f"{len(self.monitored_columns)} features."
        )

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        num_buckets: int = 10,
    ) -> float:
        """Computes Population Stability Index (PSI) between reference and current samples."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Adjust bucket count dynamically for small sample buffers to avoid sample-size noise
        adjusted_buckets = max(3, min(num_buckets, len(actual) // 20))
        percentiles = np.linspace(0, 100, adjusted_buckets + 1)
        try:
            raw_bins = np.percentile(expected, percentiles)
            bins = np.unique(raw_bins)
            if len(bins) < 2:
                bins = np.linspace(np.min(expected), np.max(expected) + 1e-5, adjusted_buckets + 1)
        except Exception:
            bins = np.linspace(np.min(expected), np.max(expected) + 1e-5, adjusted_buckets + 1)

        # Count frequencies in each bucket
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # Convert to proportions with smoothing epsilon to prevent division by zero
        eps = 1e-3
        expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
        actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

        # PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(max(0.0, psi_value))

    def evaluate_feature_drift(
        self,
        feature_name: str,
        current_samples: np.ndarray,
    ) -> FeatureDriftReport:
        """Computes PSI and KS-test for a single feature against reference distribution."""
        if self.reference_data is None or feature_name not in self.reference_data.columns:
            raise ValueError(f"Feature '{feature_name}' not available in reference data.")

        ref_series = self.reference_data[feature_name].dropna().values
        cur_series = np.asarray(current_samples, dtype=float)
        cur_series = cur_series[~np.isnan(cur_series)]

        if len(cur_series) < 10:
            return FeatureDriftReport(
                feature_name=feature_name,
                psi_score=0.0,
                ks_statistic=0.0,
                ks_p_value=1.0,
                drift_status="STABLE",
                is_drifted=False,
            )

        # 1. PSI Calculation
        psi = self.calculate_psi(ref_series, cur_series)

        # 2. Kolmogorov-Smirnov Two-Sample Test
        ks_res = ks_2samp(ref_series, cur_series)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 3. Status Classification
        if psi >= self.psi_threshold_drift:
            status = "SIGNIFICANT_DRIFT"
            is_drift = True
        elif psi >= self.psi_threshold_warning or (ks_stat > 0.15 and p_val < self.ks_alpha):
            status = "MODERATE_SHIFT"
            is_drift = False
        else:
            status = "STABLE"
            is_drift = False

        return FeatureDriftReport(
            feature_name=feature_name,
            psi_score=round(psi, 5),
            ks_statistic=round(ks_stat, 5),
            ks_p_value=round(p_val, 5),
            drift_status=status,
            is_drifted=is_drift,
        )

    def ingest_streaming_transaction(self, tx_dict: Dict[str, float]) -> None:
        """Adds incoming transaction features to the rolling drift monitoring buffer."""
        self.stream_buffer.append(tx_dict)
        if len(self.stream_buffer) > self.buffer_size:
            self.stream_buffer.pop(0)

    def evaluate_system_drift(self) -> SystemDriftSummary:
        """Evaluates drift across all features in the current streaming buffer."""
        if not self.stream_buffer or self.reference_data is None:
            return SystemDriftSummary(
                overall_system_status="HEALTHY",
                drifted_feature_count=0,
                total_features_monitored=len(self.monitored_columns),
                max_psi_feature="None",
                max_psi_value=0.0,
                recommendation="Insufficient streaming buffer data for drift analysis.",
                feature_reports=[],
            )

        buffer_df = pd.DataFrame(self.stream_buffer)
        reports: List[FeatureDriftReport] = []
        max_psi = 0.0
        max_psi_feat = "None"
        drift_count = 0

        for col in self.monitored_columns:
            if col in buffer_df.columns:
                rep = self.evaluate_feature_drift(col, buffer_df[col].values)
                reports.append(rep)
                if rep.psi_score > max_psi:
                    max_psi = rep.psi_score
                    max_psi_feat = col
                if rep.is_drifted:
                    drift_count += 1

        if drift_count >= 3 or max_psi >= self.psi_threshold_drift:
            overall = "ACTION_REQUIRED"
            rec = "Significant concept drift detected in key fraud indicators. Trigger Automated Pipeline Retraining."
        elif drift_count >= 1 or max_psi >= self.psi_threshold_warning:
            overall = "WARNING"
            rec = "Moderate distribution shifts detected. Increase manual review monitoring frequency."
        else:
            overall = "HEALTHY"
            rec = "All transaction feature distributions are stable and aligned with training baseline."

        return SystemDriftSummary(
            overall_system_status=overall,
            drifted_feature_count=drift_count,
            total_features_monitored=len(reports),
            max_psi_feature=max_psi_feat,
            max_psi_value=round(max_psi, 4),
            recommendation=rec,
            feature_reports=reports,
        )
