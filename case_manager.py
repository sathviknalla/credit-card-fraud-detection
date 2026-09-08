"""Phase 7: Fraud Case Management System.

Tracks flagged transactions through a formal investigative state machine:
    FLAGGED → UNDER_REVIEW → CONFIRMED_FRAUD | CLEARED | ESCALATED

Provides:
  - Analyst case queues prioritized by expected financial loss
  - Full audit trail logging for regulatory compliance (PCI-DSS)
  - Case statistics and workload summary
  - Thread-safe concurrent case updates

Use with the FastAPI app by importing the global CASE_MANAGER instance.
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case State Machine
# ---------------------------------------------------------------------------

class CaseStatus(str, Enum):
    """Formal state machine states for a fraud investigation case."""
    FLAGGED = "FLAGGED"               # Auto-flagged by ML model, awaiting analyst
    UNDER_REVIEW = "UNDER_REVIEW"     # Analyst has opened case
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"  # Analyst confirmed: fraud
    CLEARED = "CLEARED"               # Analyst confirmed: legitimate
    ESCALATED = "ESCALATED"           # Requires senior review or legal action


class CasePriority(str, Enum):
    """Risk-based analyst triage priority levels."""
    CRITICAL = "CRITICAL"   # Expected loss > $500 or prob > 0.90
    HIGH = "HIGH"           # Expected loss > $200 or prob > 0.75
    MEDIUM = "MEDIUM"       # Expected loss > $50  or prob > 0.50
    LOW = "LOW"             # Expected loss <= $50


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single immutable audit log entry for compliance traceability."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = ""
    from_status: Optional[str] = None
    to_status: Optional[str] = None
    analyst_id: Optional[str] = None
    notes: str = ""


@dataclass
class FraudCase:
    """Complete lifecycle record for a single fraud investigation case.

    Attributes:
        case_id: Unique identifier for this investigation case.
        transaction_id: The original transaction being investigated.
        fraud_probability: ML model's raw fraud probability score.
        expected_loss: Estimated financial exposure if fraud confirmed.
        transaction_amount: Original transaction value.
        status: Current state in the case lifecycle state machine.
        priority: Analyst triage priority level.
        created_at: UTC ISO 8601 timestamp when the case was opened.
        updated_at: UTC ISO 8601 timestamp of most recent status change.
        assigned_analyst: Analyst ID currently handling the case.
        top_risk_factors: Top SHAP feature contributions from ML explanation.
        audit_trail: Ordered list of all state transitions and analyst actions.
        metadata: Arbitrary additional context (merchant info, IP, device ID, etc.)
    """
    transaction_id: str
    fraud_probability: float
    expected_loss: float
    transaction_amount: float

    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: CaseStatus = CaseStatus.FLAGGED
    priority: CasePriority = CasePriority.MEDIUM
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    assigned_analyst: Optional[str] = None
    top_risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[AuditEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute priority from fraud probability and expected loss at creation."""
        self.priority = self._compute_priority()
        # Log initial creation event
        self.audit_trail.append(
            AuditEvent(
                action="CASE_CREATED",
                to_status=self.status.value,
                notes=f"Auto-flagged by ML model. P(Fraud)={self.fraud_probability:.4f}, "
                      f"Expected Loss=${self.expected_loss:.2f}",
            )
        )

    def _compute_priority(self) -> CasePriority:
        """Derives analyst triage priority from ML probability and expected loss."""
        if self.fraud_probability >= 0.90 or self.expected_loss >= 500:
            return CasePriority.CRITICAL
        elif self.fraud_probability >= 0.75 or self.expected_loss >= 200:
            return CasePriority.HIGH
        elif self.fraud_probability >= 0.50 or self.expected_loss >= 50:
            return CasePriority.MEDIUM
        else:
            return CasePriority.LOW

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the case to a plain dictionary for JSON responses."""
        return {
            "case_id": self.case_id,
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "fraud_probability": round(self.fraud_probability, 4),
            "expected_loss": round(self.expected_loss, 2),
            "transaction_amount": round(self.transaction_amount, 2),
            "assigned_analyst": self.assigned_analyst,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "top_risk_factors": self.top_risk_factors,
            "audit_trail_count": len(self.audit_trail),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Allowed State Transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: Dict[CaseStatus, List[CaseStatus]] = {
    CaseStatus.FLAGGED: [CaseStatus.UNDER_REVIEW, CaseStatus.CLEARED],
    CaseStatus.UNDER_REVIEW: [
        CaseStatus.CONFIRMED_FRAUD,
        CaseStatus.CLEARED,
        CaseStatus.ESCALATED,
    ],
    CaseStatus.CONFIRMED_FRAUD: [],  # Terminal state
    CaseStatus.CLEARED: [],          # Terminal state
    CaseStatus.ESCALATED: [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED],
}


# ---------------------------------------------------------------------------
# Case Manager
# ---------------------------------------------------------------------------

class FraudCaseManager:
    """Thread-safe fraud case management system.

    Maintains an in-memory registry of all fraud investigation cases with
    full audit logging, state transition validation, and prioritized analyst
    queue generation.

    Args:
        max_cases: Maximum number of cases to retain in memory before
            auto-archiving the oldest resolved cases. Default: 10,000.
    """

    _PRIORITY_ORDER = {
        CasePriority.CRITICAL: 0,
        CasePriority.HIGH: 1,
        CasePriority.MEDIUM: 2,
        CasePriority.LOW: 3,
    }

    def __init__(self, max_cases: int = 10_000):
        self._cases: Dict[str, FraudCase] = {}
        self._lock = threading.Lock()
        self.max_cases = max_cases

    def open_case(
        self,
        transaction_id: str,
        fraud_probability: float,
        expected_loss: float,
        transaction_amount: float,
        top_risk_factors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FraudCase:
        """Creates a new investigation case for a flagged transaction.

        Args:
            transaction_id: Unique ID of the transaction being investigated.
            fraud_probability: ML model's fraud probability score.
            expected_loss: Estimated financial loss (P(Fraud) * transaction value).
            transaction_amount: Original transaction amount.
            top_risk_factors: Top SHAP features from model explanation.
            metadata: Additional context (merchant, IP, device, etc.)

        Returns:
            Newly created FraudCase object.
        """
        case = FraudCase(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            expected_loss=expected_loss,
            transaction_amount=transaction_amount,
            top_risk_factors=top_risk_factors or [],
            metadata=metadata or {},
        )
        with self._lock:
            self._cases[case.case_id] = case
            self._maybe_evict_old_cases()

        logger.info(
            f"Case opened: {case.case_id} | Priority: {case.priority.value} | "
            f"P(Fraud): {fraud_probability:.4f} | Expected Loss: ${expected_loss:.2f}"
        )
        return case

    def transition_status(
        self,
        case_id: str,
        new_status: CaseStatus,
        analyst_id: Optional[str] = None,
        notes: str = "",
    ) -> FraudCase:
        """Performs a validated status transition with audit logging.

        Args:
            case_id: The ID of the case to update.
            new_status: The target CaseStatus to transition to.
            analyst_id: ID of the analyst performing the action.
            notes: Free-text notes explaining the decision.

        Returns:
            Updated FraudCase object.

        Raises:
            KeyError: If case_id does not exist.
            ValueError: If the requested transition is not allowed.
        """
        with self._lock:
            if case_id not in self._cases:
                raise KeyError(f"Case '{case_id}' not found.")

            case = self._cases[case_id]
            current_status = case.status
            allowed = _VALID_TRANSITIONS.get(current_status, [])

            if new_status not in allowed:
                raise ValueError(
                    f"Invalid transition: {current_status.value} → {new_status.value}. "
                    f"Allowed: {[s.value for s in allowed]}"
                )

            old_status = case.status
            case.status = new_status
            case.updated_at = datetime.now(timezone.utc).isoformat()
            if analyst_id:
                case.assigned_analyst = analyst_id

            # Append audit event
            case.audit_trail.append(
                AuditEvent(
                    action="STATUS_TRANSITION",
                    from_status=old_status.value,
                    to_status=new_status.value,
                    analyst_id=analyst_id,
                    notes=notes,
                )
            )

            logger.info(
                f"Case {case_id}: {old_status.value} → {new_status.value} "
                f"(Analyst: {analyst_id or 'system'})"
            )
            return case

    def get_case(self, case_id: str) -> FraudCase:
        """Retrieves a case by ID.

        Raises:
            KeyError: If case_id does not exist.
        """
        with self._lock:
            if case_id not in self._cases:
                raise KeyError(f"Case '{case_id}' not found.")
            return self._cases[case_id]

    def get_analyst_queue(
        self,
        status_filter: Optional[List[CaseStatus]] = None,
        analyst_id: Optional[str] = None,
        max_results: int = 50,
    ) -> List[FraudCase]:
        """Returns a prioritized list of cases for analyst review.

        Cases are sorted by:
          1. Priority (CRITICAL first)
          2. Expected financial loss (highest first)
          3. Creation time (oldest first)

        Args:
            status_filter: Only return cases with these statuses.
                Defaults to [FLAGGED, UNDER_REVIEW, ESCALATED].
            analyst_id: Filter to only cases assigned to this analyst.
            max_results: Maximum number of cases to return.

        Returns:
            Sorted list of FraudCase objects for analyst attention.
        """
        if status_filter is None:
            status_filter = [
                CaseStatus.FLAGGED,
                CaseStatus.UNDER_REVIEW,
                CaseStatus.ESCALATED,
            ]

        with self._lock:
            candidates = [
                c for c in self._cases.values()
                if c.status in status_filter
                and (analyst_id is None or c.assigned_analyst == analyst_id)
            ]

        candidates.sort(
            key=lambda c: (
                self._PRIORITY_ORDER[c.priority],
                -c.expected_loss,
                c.created_at,
            )
        )
        return candidates[:max_results]

    def get_statistics(self) -> Dict[str, Any]:
        """Returns a workload and outcome statistics summary.

        Returns:
            Dict with counts by status, priority distribution,
            total expected exposure, and resolution rate.
        """
        with self._lock:
            cases = list(self._cases.values())

        if not cases:
            return {"total_cases": 0, "message": "No cases in registry."}

        status_counts: Dict[str, int] = {}
        priority_counts: Dict[str, int] = {}
        total_exposure = 0.0
        confirmed_fraud_count = 0
        resolved_count = 0

        for case in cases:
            status_counts[case.status.value] = status_counts.get(case.status.value, 0) + 1
            priority_counts[case.priority.value] = priority_counts.get(case.priority.value, 0) + 1
            total_exposure += case.expected_loss
            if case.status == CaseStatus.CONFIRMED_FRAUD:
                confirmed_fraud_count += 1
            if case.status in [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED]:
                resolved_count += 1

        resolution_rate = resolved_count / len(cases) if cases else 0.0
        fraud_confirmation_rate = (
            confirmed_fraud_count / resolved_count if resolved_count > 0 else 0.0
        )

        return {
            "total_cases": len(cases),
            "by_status": status_counts,
            "by_priority": priority_counts,
            "total_expected_exposure_usd": round(total_exposure, 2),
            "confirmed_fraud_count": confirmed_fraud_count,
            "resolved_count": resolved_count,
            "resolution_rate_pct": round(100 * resolution_rate, 2),
            "fraud_confirmation_rate_pct": round(100 * fraud_confirmation_rate, 2),
        }

    def get_full_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        """Returns complete audit history for a case as a list of dicts."""
        case = self.get_case(case_id)
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "action": e.action,
                "from_status": e.from_status,
                "to_status": e.to_status,
                "analyst_id": e.analyst_id,
                "notes": e.notes,
            }
            for e in case.audit_trail
        ]

    def _maybe_evict_old_cases(self) -> None:
        """Removes oldest resolved cases when registry exceeds max_cases."""
        if len(self._cases) <= self.max_cases:
            return

        # Evict terminal (resolved) cases sorted by creation time, oldest first
        resolved = sorted(
            [c for c in self._cases.values()
             if c.status in [CaseStatus.CONFIRMED_FRAUD, CaseStatus.CLEARED]],
            key=lambda c: c.created_at,
        )
        n_to_evict = len(self._cases) - self.max_cases
        for case in resolved[:n_to_evict]:
            del self._cases[case.case_id]

        logger.debug(f"Evicted {min(n_to_evict, len(resolved))} resolved cases from memory.")

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the full case registry to disk for persistence."""
        from pathlib import Path as _Path
        target_path = _Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            joblib.dump({"cases": dict(self._cases)}, target_path)
        logger.info(f"Case registry saved to {target_path} ({len(self._cases)} cases)")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FraudCaseManager":
        """Deserializes a case registry from disk."""
        data = joblib.load(filepath)
        manager = cls()
        manager._cases = data.get("cases", {})
        logger.info(f"Loaded case registry with {len(manager._cases)} cases from {filepath}")
        return manager


# Global singleton for use across the application
CASE_MANAGER = FraudCaseManager()
