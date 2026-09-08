"""Behavioral Velocity & Anomaly Feature Store Module.

Computes real-time behavioral signals, time-delta inter-arrivals,
rolling transaction velocities (1h, 6h, 24h), and user amount deviations
for enhanced fraud detection sensitivity against bot attacks and card takeovers.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EntityState:
    """Tracks state and historical transaction events for a specific cardholder / entity."""

    transaction_history: List[Tuple[float, float]] = field(default_factory=list)  # (time_sec, amount)
    last_transaction_time: Optional[float] = None
    total_spend: float = 0.0
    transaction_count: int = 0


class BehavioralVelocityEngine:
    """Computes streaming and batch velocity metrics over temporal windows.

    Derived Signals:
      - `time_delta_seconds`: Elapsed time since the previous transaction.
      - `tx_velocity_1h`: Number of transactions in the trailing 1-hour window.
      - `tx_velocity_6h`: Number of transactions in the trailing 6-hour window.
      - `tx_velocity_24h`: Number of transactions in the trailing 24-hour window.
      - `amount_to_avg_ratio`: Ratio of current amount to the rolling historical average.
      - `amount_z_score`: Standardized deviation from the cardholder's historical amount mean.
    """

    def __init__(self, default_entity_id: str = "global_stream"):
        self.default_entity_id = default_entity_id
        self.entity_registry: Dict[str, EntityState] = {}

    def get_or_create_entity(self, entity_id: str) -> EntityState:
        """Retrieves or initializes an entity's historical state."""
        if entity_id not in self.entity_registry:
            self.entity_registry[entity_id] = EntityState()
        return self.entity_registry[entity_id]

    def compute_single_features(
        self,
        time_sec: float,
        amount: float,
        entity_id: Optional[str] = None,
        update_state: bool = True,
    ) -> Dict[str, float]:
        """Computes real-time behavioral velocity features for a single incoming transaction event.

        Args:
            time_sec: Transaction timestamp in elapsed seconds.
            amount: Monetary value of the transaction.
            entity_id: Card / user identifier. Defaults to global streaming queue.
            update_state: Whether to record this transaction into the entity's history.

        Returns:
            Dictionary of computed velocity and anomaly features.
        """
        eid = entity_id or self.default_entity_id
        state = self.get_or_create_entity(eid)

        # 1. Time-delta calculation
        if state.last_transaction_time is not None:
            time_delta = max(0.0, time_sec - state.last_transaction_time)
        else:
            time_delta = 86400.0  # Default 24h for initial transaction

        # 2. Rolling window counts
        h1_cutoff = time_sec - 3600.0
        h6_cutoff = time_sec - 21600.0
        h24_cutoff = time_sec - 86400.0

        v1h = sum(1 for t, _ in state.transaction_history if t >= h1_cutoff)
        v6h = sum(1 for t, _ in state.transaction_history if t >= h6_cutoff)
        v24h = sum(1 for t, _ in state.transaction_history if t >= h24_cutoff)

        # 3. Historical Amount Statistics
        if state.transaction_history:
            recent_amounts = [a for _, a in state.transaction_history]
            avg_amount = float(np.mean(recent_amounts))
            std_amount = float(np.std(recent_amounts)) if len(recent_amounts) > 1 else 10.0
            std_amount = max(1.0, std_amount)  # Avoid division by zero

            amount_ratio = amount / max(1.0, avg_amount)
            amount_z = (amount - avg_amount) / std_amount
        else:
            amount_ratio = 1.0
            amount_z = 0.0

        if update_state:
            # Append new transaction
            state.transaction_history.append((time_sec, amount))
            state.last_transaction_time = time_sec
            state.total_spend += amount
            state.transaction_count += 1

            # Prune events older than 48 hours to conserve memory
            prune_cutoff = time_sec - 172800.0
            state.transaction_history = [
                (t, a) for t, a in state.transaction_history if t >= prune_cutoff
            ]

        return {
            "time_delta_seconds": float(time_delta),
            "tx_velocity_1h": float(v1h),
            "tx_velocity_6h": float(v6h),
            "tx_velocity_24h": float(v24h),
            "amount_to_avg_ratio": float(amount_ratio),
            "amount_z_score": float(np.clip(amount_z, -10.0, 10.0)),
        }

    def compute_batch_features(
        self,
        df: pd.DataFrame,
        time_col: str = "Time",
        amount_col: str = "Amount",
        entity_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Vectorized/sequential batch generation of velocity features across a DataFrame."""
        df_sorted = df.sort_values(by=time_col).copy()
        n_rows = len(df_sorted)

        time_deltas = np.zeros(n_rows, dtype=np.float32)
        v1h = np.zeros(n_rows, dtype=np.float32)
        v6h = np.zeros(n_rows, dtype=np.float32)
        v24h = np.zeros(n_rows, dtype=np.float32)
        ratios = np.ones(n_rows, dtype=np.float32)
        z_scores = np.zeros(n_rows, dtype=np.float32)

        # Temporary engine for batch computation
        temp_engine = BehavioralVelocityEngine()

        for i in range(n_rows):
            row_time = float(df_sorted.iloc[i][time_col])
            row_amt = float(df_sorted.iloc[i][amount_col])
            eid = str(df_sorted.iloc[i][entity_col]) if entity_col and entity_col in df_sorted else None

            feats = temp_engine.compute_single_features(row_time, row_amt, entity_id=eid)
            time_deltas[i] = feats["time_delta_seconds"]
            v1h[i] = feats["tx_velocity_1h"]
            v6h[i] = feats["tx_velocity_6h"]
            v24h[i] = feats["tx_velocity_24h"]
            ratios[i] = feats["amount_to_avg_ratio"]
            z_scores[i] = feats["amount_z_score"]

        df_sorted["time_delta_seconds"] = time_deltas
        df_sorted["tx_velocity_1h"] = v1h
        df_sorted["tx_velocity_6h"] = v6h
        df_sorted["tx_velocity_24h"] = v24h
        df_sorted["amount_to_avg_ratio"] = ratios
        df_sorted["amount_z_score"] = z_scores

        return df_sorted
