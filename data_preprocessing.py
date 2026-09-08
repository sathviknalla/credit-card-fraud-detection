"""Data Preprocessing & Validation Split Module for Financial Fraud Detection.

Implements strict leakage-safe time-based splitting, robust feature scaling,
feature engineering, and configurable class imbalance handling (SMOTE,
class-weighting, and hybrid resampling).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import CONFIG, DataConfig, ImbalanceConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class DatasetSplits:
    """Container holding strictly partitioned training, validation, and test sets."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: List[str]


class TemporalDataSplitter:
    """Enforces strict, leakage-safe chronological partitioning of transaction data.

    In financial fraud detection, future information cannot leak into past observations.
    Standard random splits violate causality and create optimistic performance bias.
    This splitter orders transactions chronologically and splits sequentially:
      - Train set (earliest T% of time)
      - Validation set (next V% of time for threshold & hyperparameter tuning)
      - Out-of-Time Test set (final E% of time for final unbiased evaluation)
    """

    def __init__(self, config: DataConfig = CONFIG.data):
        self.config = config

    def split(self, df: pd.DataFrame) -> DatasetSplits:
        """Splits DataFrame into train, val, and test partitions based on Time order.

        Args:
            df: Input DataFrame containing features, 'Time', and 'Class' target.

        Returns:
            DatasetSplits object with partitioned X and y subsets.
        """
        logger.info(
            f"Splitting dataset of {len(df):,} rows using chronological order on column: {self.config.time_column}"
        )

        # 1. Sort strictly by timestamp/time offset to preserve chronological timeline
        sorted_df = df.sort_values(by=self.config.time_column).reset_index(drop=True)

        total_rows = len(sorted_df)
        train_end_idx = int(total_rows * self.config.train_ratio)
        val_end_idx = int(total_rows * (self.config.train_ratio + self.config.val_ratio))

        train_slice = sorted_df.iloc[:train_end_idx]
        val_slice = sorted_df.iloc[train_end_idx:val_end_idx]
        test_slice = sorted_df.iloc[val_end_idx:]

        target_col = self.config.target_column
        feature_cols = [col for col in sorted_df.columns if col != target_col]

        splits = DatasetSplits(
            X_train=train_slice[feature_cols].copy(),
            y_train=train_slice[target_col].copy(),
            X_val=val_slice[feature_cols].copy(),
            y_val=val_slice[target_col].copy(),
            X_test=test_slice[feature_cols].copy(),
            y_test=test_slice[target_col].copy(),
            feature_names=feature_cols,
        )

        logger.info(
            f"Train Set: {len(splits.X_train):,} samples | Fraud Rate: {splits.y_train.mean():.4%}"
        )
        logger.info(
            f"Val Set:   {len(splits.X_val):,} samples | Fraud Rate: {splits.y_val.mean():.4%}"
        )
        logger.info(
            f"Test Set:  {len(splits.X_test):,} samples | Fraud Rate: {splits.y_test.mean():.4%}"
        )

        return splits


class LeakageSafePreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe transformer that scales Amount and engineered features.

    Fitting is strictly performed on training data only. Validation and Test
    partitions are transformed using the parameters learned from Train.
    """

    def __init__(self, time_col: str = "Time", amount_col: str = "Amount"):
        self.time_col = time_col
        self.amount_col = amount_col
        self.amount_scaler = RobustScaler()
        self.hour_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns_: List[str] = []

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derives domain-specific fraud features from raw transaction fields."""
        df = X.copy()

        # 1. Cyclical Hour of Day feature from elapsed seconds in Time column
        if self.time_col in df.columns:
            # 86400 seconds in a day
            hours_elapsed = (df[self.time_col] / 3600.0) % 24.0
            df["hour_sin"] = np.sin(2 * np.pi * hours_elapsed / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hours_elapsed / 24.0)
            df["time_hour"] = hours_elapsed

        # 2. Log-transformed and scaled Amount to handle extreme transaction values
        if self.amount_col in df.columns:
            # Add small epsilon to prevent log(0)
            df["log_amount"] = np.log1p(np.maximum(0, df[self.amount_col]))

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LeakageSafePreprocessor":
        """Fits scalers only on training feature distributions."""
        df_eng = self._engineer_features(X)

        if self.amount_col in df_eng.columns:
            self.amount_scaler.fit(df_eng[[self.amount_col, "log_amount"]])

        if "time_hour" in df_eng.columns:
            self.hour_scaler.fit(df_eng[["time_hour"]])

        transformed_df = self._transform_core(df_eng)
        self.feature_columns_ = list(transformed_df.columns)
        self.is_fitted = True
        return self

    def _transform_core(self, df_eng: pd.DataFrame) -> pd.DataFrame:
        """Applies fitted transformation to engineered feature DataFrame."""
        df = df_eng.copy()

        if self.amount_col in df.columns:
            scaled_amounts = self.amount_scaler.transform(df[[self.amount_col, "log_amount"]])
            df[f"scaled_{self.amount_col}"] = scaled_amounts[:, 0]
            df["scaled_log_amount"] = scaled_amounts[:, 1]
            df.drop(columns=[self.amount_col, "log_amount"], inplace=True)

        if "time_hour" in df.columns:
            df["scaled_time_hour"] = self.hour_scaler.transform(df[["time_hour"]])[:, 0]
            df.drop(columns=["time_hour"], inplace=True)

        # Drop original raw Time column after feature extraction
        if self.time_col in df.columns:
            df.drop(columns=[self.time_col], inplace=True)

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using strictly fitted training parameters."""
        if not self.is_fitted:
            raise RuntimeError("LeakageSafePreprocessor must be fitted before calling transform().")
        df_eng = self._engineer_features(X)
        transformed = self._transform_core(df_eng)

        # Ensure consistent column ordering matching fit time
        return transformed.reindex(columns=self.feature_columns_, fill_value=0.0)


class ImbalanceHandler:
    """Encapsulates configurable resampling techniques for severe class imbalance.

    NOTE: Resampling is strictly applied to the TRAINING SET ONLY.
    Never resample validation or test sets to prevent data leakage and artificial distributions.
    """

    def __init__(self, config: ImbalanceConfig = CONFIG.imbalance):
        self.config = config

    def handle_imbalance(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[float]]:
        """Applies configured resampling strategy on training data.

        Returns:
            Tuple of (Resampled X_train, Resampled y_train, scale_pos_weight).
        """
        strategy = self.config.strategy.lower()
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = float(neg_count / max(1, pos_count))

        logger.info(
            f"Applying imbalance handling strategy: '{strategy}'. "
            f"Original Train: {neg_count:,} Legitimate, {pos_count:,} Fraud."
        )

        if strategy == "none":
            return X_train, y_train, 1.0

        if strategy == "class_weight":
            # Return original data along with calculated scale_pos_weight for loss function
            return X_train, y_train, scale_pos_weight

        if strategy == "smote":
            smote = SMOTE(
                sampling_strategy=self.config.smote_sampling_strategy,
                k_neighbors=self.config.smote_k_neighbors,
                random_state=CONFIG.data.random_state,
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            logger.info(
                f"Post-SMOTE: {(y_res == 0).sum():,} Legitimate, {(y_res == 1).sum():,} Fraud."
            )
            return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res), 1.0

        if strategy == "hybrid":
            # SMOTE to boost minority class, followed by gentle RandomUnderSampler
            smote = SMOTE(
                sampling_strategy=self.config.smote_sampling_strategy,
                k_neighbors=self.config.smote_k_neighbors,
                random_state=CONFIG.data.random_state,
            )
            X_sm, y_sm = smote.fit_resample(X_train, y_train)

            under = RandomUnderSampler(
                sampling_strategy=self.config.undersample_ratio or 0.2,
                random_state=CONFIG.data.random_state,
            )
            X_res, y_res = under.fit_resample(X_sm, y_sm)
            logger.info(
                f"Post-Hybrid Resampling: {(y_res == 0).sum():,} Legitimate, {(y_res == 1).sum():,} Fraud."
            )
            return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res), 1.0

        raise ValueError(f"Unknown imbalance handling strategy: {strategy}")


def generate_synthetic_fraud_dataset(
    n_samples: int = 284807,
    fraud_rate: float = 0.00172,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates a realistic synthetic credit card transaction dataset matching the Kaggle schema.

    Used for testing and initialization when the raw CSV file is not present locally.
    Generates 28 anonymized PCA features (V1..V28), Time (seconds), Amount ($), and Class (0/1).
    """
    logger.info(
        f"Generating synthetic credit card dataset ({n_samples:,} transactions, fraud rate={fraud_rate:.3%})..."
    )
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # 1. Simulate Time column (monotonically increasing over 2 days = 172800 seconds)
    time_series = np.sort(rng.uniform(0, 172800, size=n_samples))

    # 2. Simulate 28 PCA features (V1 to V28)
    # Legit transactions have standard normal distributions
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud transactions have shifted means on key known discriminative features (e.g., V4, V11, V12, V14, V17)
    v_fraud = rng.normal(loc=0.0, scale=1.5, size=(n_fraud, 28))
    v_fraud[:, 3] += 2.5   # V4 higher in fraud
    v_fraud[:, 10] += 2.0  # V11 higher in fraud
    v_fraud[:, 11] -= 3.0  # V12 lower in fraud
    v_fraud[:, 13] -= 3.5  # V14 significantly lower in fraud
    v_fraud[:, 16] -= 2.8  # V17 lower in fraud

    # 3. Simulate Amount (log-normal distribution with heavier tail for legit and distinct clusters for fraud)
    amt_legit = np.clip(rng.lognormal(mean=3.0, sigma=1.2, size=n_legit), 0.5, 5000.0)
    amt_fraud = np.clip(rng.lognormal(mean=4.2, sigma=1.5, size=n_fraud), 1.0, 3000.0)

    # Combine into unified dataset
    legit_df = pd.DataFrame(v_legit, columns=[f"V{i}" for i in range(1, 29)])
    legit_df["Amount"] = amt_legit
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(v_fraud, columns=[f"V{i}" for i in range(1, 29)])
    fraud_df["Amount"] = amt_fraud
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], axis=0, ignore_index=True)
    # Shuffle slightly within local temporal blocks to assign realistic timestamps
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["Time"] = time_series

    # Reorder columns to standard schema: Time, V1..V28, Amount, Class
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    return df[cols]
