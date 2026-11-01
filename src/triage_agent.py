"""
Deterministic rule-based triage agent for fleet maintenance decisions.
Maps predicted Remaining Useful Life and anomaly indicators to transparent,
auditable priority tiers with escalation rules and defensive data safeguards.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

PRIORITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class TriageDecision:
    """Structured decision output produced by the deterministic triage rules."""
    engine_id: int
    cycle: int
    predicted_rul: Optional[float]
    anomaly_score: float
    anomaly_status: str
    priority: str
    reason: str
    recommended_action: str
    should_create_work_order: bool
    work_order_id: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cycle": self.cycle,
            "predicted_rul": self.predicted_rul,
            "anomaly_score": self.anomaly_score,
            "anomaly_status": self.anomaly_status,
            "priority": self.priority,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
            "should_create_work_order": self.should_create_work_order,
            "work_order_id": self.work_order_id,
            "created_at": self.created_at,
        }


class RuleBasedTriageAgent:
    """
    Transparent, explainable triage agent executing deterministic maintenance policies.
    Guarantees that no black-box decisions or LLM hallucinations enter physical safety workflows.
    """

    def __init__(
        self,
        critical_rul_threshold: float = 15.0,
        high_rul_threshold: float = 30.0,
        medium_rul_threshold: float = 60.0,
        anomaly_escalation_enabled: bool = True,
        severe_anomaly_threshold: float = 0.70,
        auto_work_order_priorities: Optional[List[str]] = None,
    ):
        self.critical_rul = critical_rul_threshold
        self.high_rul = high_rul_threshold
        self.medium_rul = medium_rul_threshold
        self.anomaly_escalation_enabled = anomaly_escalation_enabled
        self.severe_anomaly_threshold = severe_anomaly_threshold
        self.auto_work_order_priorities = auto_work_order_priorities or ["CRITICAL", "HIGH", "MEDIUM"]

    def evaluate_engine(
        self,
        engine_id: int,
        cycle: int,
        predicted_rul: Optional[float],
        anomaly_score: float = 0.0,
        anomaly_status: str = "NORMAL",
        data_valid: bool = True,
        validation_error: Optional[str] = None,
    ) -> TriageDecision:
        """
        Evaluate engine state against deterministic rules.

        Safety Rules:
        1. Invalid/missing data or failed inference triggers immediate CRITICAL investigation.
        2. Base priority evaluated against RUL thresholds.
        3. Severe anomaly detection raises priority by one level (subject to safeguards).
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_eng_id = int(engine_id)
        clean_cycle = int(cycle)

        # Safeguard 1: Data quality check
        if not data_valid:
            priority = "CRITICAL"
            reason = f"Data Quality Fault: Telemetry invalid or corrupted ({validation_error or 'Schema error'}). Immediate sensor inspection required."
            action = "Dispatch electrical and avionics technician to verify sensor harness, bus line integrity, and telemetry transceiver."
            wo_id = f"WO-ENG{clean_eng_id:03d}-C{clean_cycle:03d}-DATAFAIL"
            return TriageDecision(
                engine_id=clean_eng_id,
                cycle=clean_cycle,
                predicted_rul=None,
                anomaly_score=float(anomaly_score),
                anomaly_status="INVALID_DATA",
                priority=priority,
                reason=reason,
                recommended_action=action,
                should_create_work_order=True,
                work_order_id=wo_id,
                created_at=now_iso,
            )

        # Safeguard 2: Missing or invalid RUL prediction
        if predicted_rul is None or not (isinstance(predicted_rul, (int, float)) and predicted_rul == predicted_rul):
            priority = "CRITICAL"
            reason = "Inference Exception: Predictive model failed to generate RUL estimate. Fail-safe trigger activated."
            action = "Perform emergency diagnostic query on machine telemetry and inspect physical engine prior to next flight sortie."
            wo_id = f"WO-ENG{clean_eng_id:03d}-C{clean_cycle:03d}-INFERR"
            return TriageDecision(
                engine_id=clean_eng_id,
                cycle=clean_cycle,
                predicted_rul=None,
                anomaly_score=float(anomaly_score),
                anomaly_status=anomaly_status,
                priority=priority,
                reason=reason,
                recommended_action=action,
                should_create_work_order=True,
                work_order_id=wo_id,
                created_at=now_iso,
            )

        rul_val = float(predicted_rul)

        # Base Rule Evaluation based on RUL
        if rul_val <= self.critical_rul:
            base_priority = "CRITICAL"
            base_reason = f"Predicted RUL ({rul_val:.1f} cycles) is at or below Critical threshold ({self.critical_rul:.1f} cycles)."
        elif rul_val <= self.high_rul:
            base_priority = "HIGH"
            base_reason = f"Predicted RUL ({rul_val:.1f} cycles) is between High ({self.high_rul:.1f}) and Critical ({self.critical_rul:.1f}) thresholds."
        elif rul_val <= self.medium_rul:
            base_priority = "MEDIUM"
            base_reason = f"Predicted RUL ({rul_val:.1f} cycles) is between Medium ({self.medium_rul:.1f}) and High ({self.high_rul:.1f}) thresholds."
        else:
            base_priority = "LOW"
            base_reason = f"Predicted RUL ({rul_val:.1f} cycles) is above Medium threshold ({self.medium_rul:.1f} cycles). Nominal lifespan remains."

        final_priority = base_priority
        escalation_note = ""

        # Anomaly Escalation Rule
        is_severe_anomaly = (anomaly_score >= self.severe_anomaly_threshold) or (anomaly_status == "ANOMALOUS")
        if self.anomaly_escalation_enabled and is_severe_anomaly:
            curr_idx = PRIORITY_LEVELS.index(base_priority)
            if curr_idx < len(PRIORITY_LEVELS) - 1:
                final_priority = PRIORITY_LEVELS[curr_idx + 1]
                escalation_note = (
                    f" [ESCALATED from {base_priority} to {final_priority} due to severe anomaly "
                    f"(score={anomaly_score:.2f} >= threshold {self.severe_anomaly_threshold:.2f})]."
                )
            else:
                escalation_note = f" [Confirmed severe anomaly score {anomaly_score:.2f} at maximum priority CRITICAL]."

        full_reason = base_reason + escalation_note

        # Map to realistic advisory maintenance actions
        if final_priority == "CRITICAL":
            action = (
                "SCHEDULE IMMEDIATE INSPECTION: Remove engine from active flight roster within 24 hours. "
                "Conduct borescope examination of High Pressure Compressor (HPC), verify turbine clearances, "
                "and prepare module replacement."
            )
        elif final_priority == "HIGH":
            action = (
                "PRIORITY MAINTENANCE WINDOW: Schedule line maintenance inspection at next overnight turnaround. "
                "Inspect bleed valves, perform acoustic vibration survey, and monitor lubrication debris sensors."
            )
        elif final_priority == "MEDIUM":
            action = (
                "PLANNED DEPOT SERVICE: Flag engine for comprehensive diagnostics during next scheduled 50-cycle service interval. "
                "Track sensor trending and temperature margins."
            )
        else:
            action = (
                "ROUTINE MONITORING: Machine operating within acceptable bounds. Continue automated telemetry polling."
            )

        should_create_wo = final_priority in self.auto_work_order_priorities
        wo_id = f"WO-ENG{clean_eng_id:03d}-C{clean_cycle:03d}-{final_priority[:4]}"

        return TriageDecision(
            engine_id=clean_eng_id,
            cycle=clean_cycle,
            predicted_rul=rul_val,
            anomaly_score=float(anomaly_score),
            anomaly_status=anomaly_status,
            priority=final_priority,
            reason=full_reason,
            recommended_action=action,
            should_create_work_order=should_create_wo,
            work_order_id=wo_id,
            created_at=now_iso,
        )
