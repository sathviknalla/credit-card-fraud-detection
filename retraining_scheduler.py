"""Phase 8: Automated Retraining Trigger & Model Registry.

Monitors drift signals and PR-AUC degradation to automatically trigger
retraining. Maintains a model version registry with full experiment metadata
for rollback and champion-challenger A/B testing.

Components:
  - ModelRegistry: JSON-backed version history of all trained models.
  - RetrainingScheduler: Watches drift + PR-AUC decay → triggers retraining.
  - ChampionChallengerEvaluator: A/B comparison between two model versions.

Trigger Conditions:
  1. Drift monitor reports ACTION_REQUIRED status, OR
  2. Online learner running loss exceeds degradation_threshold, OR
  3. Manual retraining requested via API.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from config import CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """Metadata record for a single trained model version.

    Stored in the registry to track performance history and enable rollback.
    """
    version_id: str
    model_type: str
    training_timestamp: str
    pr_auc_val: float
    pr_auc_test: float
    optimal_threshold: float
    imbalance_strategy: str
    n_training_samples: int
    trigger_reason: str          # "MANUAL" | "DRIFT_DETECTED" | "LOSS_DEGRADATION"
    artifact_path: str
    is_champion: bool = False    # True = currently deployed / active model
    feature_hash: str = ""       # Hash of feature names for schema validation
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Persistent JSON-backed model version registry.

    Tracks all trained model versions with their performance metrics,
    enables rollback to previous versions, and marks the active champion.

    Args:
        registry_path: Path to the JSON registry file. Created if missing.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or (CONFIG.artifact_dir / "model_registry.json")
        self._versions: List[ModelVersion] = []
        self._lock = threading.Lock()
        self._load_from_disk()

    def register_version(self, version: ModelVersion) -> None:
        """Adds a new model version to the registry and persists to disk.

        If is_champion is True, automatically demotes any existing champion.
        """
        with self._lock:
            if version.is_champion:
                for v in self._versions:
                    v.is_champion = False

            self._versions.append(version)
            self._persist()
        logger.info(
            f"Registered model version: {version.version_id} | "
            f"PR-AUC(val): {version.pr_auc_val:.5f} | "
            f"Champion: {version.is_champion}"
        )

    def set_champion(self, version_id: str) -> ModelVersion:
        """Promotes a specific version to champion (currently deployed model).

        Demotes all other versions.

        Args:
            version_id: ID of the version to promote.

        Returns:
            The promoted ModelVersion.

        Raises:
            KeyError: If version_id is not found in the registry.
        """
        with self._lock:
            target = None
            for v in self._versions:
                v.is_champion = False
                if v.version_id == version_id:
                    target = v

            if target is None:
                raise KeyError(f"Version '{version_id}' not found in registry.")

            target.is_champion = True
            self._persist()
        logger.info(f"Champion set to: {version_id}")
        return target

    def get_champion(self) -> Optional[ModelVersion]:
        """Returns the currently active champion model version."""
        with self._lock:
            champions = [v for v in self._versions if v.is_champion]
        return champions[-1] if champions else None

    def get_best_version(self, metric: str = "pr_auc_val") -> Optional[ModelVersion]:
        """Returns the model version with the highest score on a given metric."""
        with self._lock:
            candidates = [v for v in self._versions if hasattr(v, metric)]
        if not candidates:
            return None
        return max(candidates, key=lambda v: getattr(v, metric, 0.0))

    def list_versions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent model versions as a list of dicts, newest first."""
        with self._lock:
            sorted_versions = sorted(self._versions, key=lambda v: v.training_timestamp, reverse=True)
        return [v.to_dict() for v in sorted_versions[:limit]]

    def rollback_to(self, version_id: str) -> ModelVersion:
        """Promotes an older version to champion, effectively rolling back.

        Args:
            version_id: ID of the version to restore as champion.

        Returns:
            The restored ModelVersion.
        """
        logger.warning(f"Rolling back champion model to version: {version_id}")
        return self.set_champion(version_id)

    def _load_from_disk(self) -> None:
        """Loads registry from JSON file if it exists."""
        if not self.registry_path.exists():
            logger.info(f"No existing registry found at {self.registry_path}. Starting fresh.")
            return

        try:
            with open(self.registry_path, "r") as f:
                raw = json.load(f)
            self._versions = [ModelVersion(**entry) for entry in raw.get("versions", [])]
            logger.info(f"Loaded model registry with {len(self._versions)} versions.")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}. Starting fresh.")
            self._versions = []

    def _persist(self) -> None:
        """Persists the full registry to JSON file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"versions": [v.to_dict() for v in self._versions]}
        with open(self.registry_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Champion-Challenger Evaluator
# ---------------------------------------------------------------------------

@dataclass
class ABTestResult:
    """Result of a champion-challenger A/B comparison."""
    champion_version_id: str
    challenger_version_id: str
    champion_pr_auc: float
    challenger_pr_auc: float
    winner: str              # "champion" | "challenger" | "tie"
    delta_pr_auc: float
    promote_challenger: bool
    evaluation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        return (
            f"A/B Test Result\n"
            f"  Champion  [{self.champion_version_id[:8]}] PR-AUC: {self.champion_pr_auc:.5f}\n"
            f"  Challenger[{self.challenger_version_id[:8]}] PR-AUC: {self.challenger_pr_auc:.5f}\n"
            f"  Delta: {self.delta_pr_auc:+.5f} | Winner: {self.winner.upper()}\n"
            f"  Promote Challenger: {'YES ✅' if self.promote_challenger else 'NO ❌'}"
        )


class ChampionChallengerEvaluator:
    """Evaluates a challenger model against the current champion.

    Promotes the challenger only if it improves PR-AUC by at least
    min_improvement_threshold, preventing frequent unnecessary swaps.

    Args:
        min_improvement_threshold: Minimum PR-AUC improvement required
            to promote challenger to champion. Default: 0.005 (0.5%).
    """

    def __init__(self, min_improvement_threshold: float = 0.005):
        self.min_improvement_threshold = min_improvement_threshold

    def evaluate(
        self,
        champion_probas: np.ndarray,
        challenger_probas: np.ndarray,
        y_true: np.ndarray,
        champion_version_id: str,
        challenger_version_id: str,
    ) -> ABTestResult:
        """Compares champion and challenger on the same labeled test set.

        Args:
            champion_probas: Probability predictions from the champion model.
            challenger_probas: Probability predictions from the challenger model.
            y_true: Ground truth labels.
            champion_version_id: Registry version ID for the champion.
            challenger_version_id: Registry version ID for the challenger.

        Returns:
            ABTestResult with comparison details and promotion recommendation.
        """
        champion_auc = float(average_precision_score(y_true, champion_probas))
        challenger_auc = float(average_precision_score(y_true, challenger_probas))
        delta = challenger_auc - champion_auc
        promote = delta >= self.min_improvement_threshold

        if abs(delta) < 0.001:
            winner = "tie"
        elif challenger_auc > champion_auc:
            winner = "challenger"
        else:
            winner = "champion"

        result = ABTestResult(
            champion_version_id=champion_version_id,
            challenger_version_id=challenger_version_id,
            champion_pr_auc=champion_auc,
            challenger_pr_auc=challenger_auc,
            winner=winner,
            delta_pr_auc=round(delta, 6),
            promote_challenger=promote,
        )
        logger.info(f"\n{result}")
        return result


# ---------------------------------------------------------------------------
# Retraining Scheduler
# ---------------------------------------------------------------------------

@dataclass
class RetrainingTrigger:
    """Records what triggered an automatic retraining event."""
    reason: str
    drift_status: Optional[str] = None
    running_loss: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RetrainingScheduler:
    """Automated retraining orchestrator.

    Monitors drift and loss signals in a background thread and triggers
    the training pipeline when thresholds are breached. Updates the model
    registry and auto-promotes champions based on A/B test results.

    Args:
        registry: ModelRegistry to log new versions into.
        check_interval_seconds: How often to poll drift/loss signals.
        loss_degradation_threshold: Running log-loss above which to retrain.
        auto_promote: If True, automatically promote challenger to champion
            when it wins the A/B test. If False, requires manual promotion.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        check_interval_seconds: int = 300,
        loss_degradation_threshold: float = 0.35,
        auto_promote: bool = True,
    ):
        self.registry = registry
        self.check_interval_seconds = check_interval_seconds
        self.loss_degradation_threshold = loss_degradation_threshold
        self.auto_promote = auto_promote

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running: bool = False
        self._trigger_history: List[RetrainingTrigger] = []
        self._retraining_count: int = 0

    def start(self) -> None:
        """Starts the background monitoring thread."""
        if self._is_running:
            logger.warning("RetrainingScheduler is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        logger.info(
            f"RetrainingScheduler started. Poll interval: {self.check_interval_seconds}s"
        )

    def stop(self) -> None:
        """Gracefully stops the background monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._is_running = False
        logger.info("RetrainingScheduler stopped.")

    def trigger_manual(self, reason: str = "MANUAL") -> RetrainingTrigger:
        """Manually triggers an immediate retraining cycle.

        Args:
            reason: Human-readable description of why retraining was triggered.

        Returns:
            RetrainingTrigger record documenting the event.
        """
        trigger = RetrainingTrigger(reason=reason)
        self._execute_retraining(trigger)
        return trigger

    def get_status(self) -> Dict[str, Any]:
        """Returns the current scheduler status and trigger history."""
        return {
            "is_running": self._is_running,
            "retraining_count": self._retraining_count,
            "check_interval_seconds": self.check_interval_seconds,
            "loss_degradation_threshold": self.loss_degradation_threshold,
            "auto_promote": self.auto_promote,
            "recent_triggers": [
                {
                    "reason": t.reason,
                    "drift_status": t.drift_status,
                    "running_loss": t.running_loss,
                    "timestamp": t.timestamp,
                }
                for t in self._trigger_history[-5:]
            ],
        }

    def _monitor_loop(self) -> None:
        """Internal monitoring loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                self._check_and_trigger()
            except Exception as e:
                logger.error(f"RetrainingScheduler monitoring error: {e}", exc_info=True)

            self._stop_event.wait(timeout=self.check_interval_seconds)

    def _check_and_trigger(self) -> None:
        """Evaluates trigger conditions and fires retraining if needed."""
        # Import lazily to avoid circular imports at module load time
        from fastapi_app import STATE

        trigger: Optional[RetrainingTrigger] = None

        # Condition 1: Drift signal
        if STATE.drift_monitor is not None:
            try:
                drift_summary = STATE.drift_monitor.evaluate_system_drift()
                if drift_summary.overall_system_status == "ACTION_REQUIRED":
                    trigger = RetrainingTrigger(
                        reason="DRIFT_DETECTED",
                        drift_status=drift_summary.overall_system_status,
                    )
                    logger.warning(
                        "Drift ACTION_REQUIRED threshold breached. Triggering retraining..."
                    )
            except Exception as e:
                logger.debug(f"Drift check skipped: {e}")

        # Condition 2: Online learner loss degradation
        if trigger is None and STATE.online_learner is not None:
            running_loss = STATE.online_learner.stats.running_loss
            if running_loss > self.loss_degradation_threshold:
                trigger = RetrainingTrigger(
                    reason="LOSS_DEGRADATION",
                    running_loss=running_loss,
                )
                logger.warning(
                    f"Online learner loss {running_loss:.4f} exceeded threshold "
                    f"{self.loss_degradation_threshold:.4f}. Triggering retraining..."
                )

        if trigger is not None:
            self._execute_retraining(trigger)

    def _execute_retraining(self, trigger: RetrainingTrigger) -> None:
        """Runs the training pipeline, evaluates A/B vs champion, updates registry."""
        logger.info(f"Executing retraining. Trigger: {trigger.reason}")
        self._trigger_history.append(trigger)

        try:
            from train import run_training_pipeline

            new_model = run_training_pipeline(
                model_type="xgboost",
                imbalance_strategy="hybrid",
                n_samples=50000,
            )
            self._retraining_count += 1

            # Load threshold metadata for new model PR-AUC
            threshold_path = CONFIG.artifact_dir / CONFIG.threshold_artifact_name
            pr_auc_val = 0.0
            pr_auc_test = 0.0
            optimal_threshold = 0.5
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    import json
                    meta = json.load(f)
                    pr_auc_val = meta.get("pr_auc", 0.0)
                    pr_auc_test = meta.get("test_pr_auc", 0.0)
                    optimal_threshold = meta.get("optimal_threshold", 0.5)

            import uuid as _uuid
            version_id = str(_uuid.uuid4())[:8]
            new_version = ModelVersion(
                version_id=version_id,
                model_type=new_model.model_name,
                training_timestamp=datetime.now(timezone.utc).isoformat(),
                pr_auc_val=pr_auc_val,
                pr_auc_test=pr_auc_test,
                optimal_threshold=optimal_threshold,
                imbalance_strategy="hybrid",
                n_training_samples=50000,
                trigger_reason=trigger.reason,
                artifact_path=str(CONFIG.artifact_dir / CONFIG.model_artifact_name),
                is_champion=False,
            )

            current_champion = self.registry.get_champion()

            if current_champion is None:
                # No existing champion — promote immediately
                new_version.is_champion = True
                self.registry.register_version(new_version)
                logger.info(f"No prior champion. New version {version_id} promoted to champion.")
            else:
                self.registry.register_version(new_version)
                logger.info(
                    f"New version {version_id} registered. "
                    f"Champion is {current_champion.version_id}. "
                    f"PR-AUC challenger={pr_auc_val:.5f} vs champion={current_champion.pr_auc_val:.5f}"
                )
                if self.auto_promote and pr_auc_val >= current_champion.pr_auc_val + 0.005:
                    self.registry.set_champion(version_id)
                    logger.info(f"Auto-promoted challenger {version_id} to champion.")

        except Exception as e:
            logger.error(f"Retraining execution failed: {e}", exc_info=True)


# Global singleton instances
REGISTRY = ModelRegistry()
SCHEDULER = RetrainingScheduler(registry=REGISTRY)
AB_EVALUATOR = ChampionChallengerEvaluator()
