"""
Unit tests for deterministic rule-based triage agent.
Tests threshold classification, priority escalation, and fail-safe guards.
"""

import pytest
from src.triage_agent import RuleBasedTriageAgent


@pytest.fixture
def agent():
    return RuleBasedTriageAgent(
        critical_rul_threshold=15.0,
        high_rul_threshold=30.0,
        medium_rul_threshold=60.0,
        anomaly_escalation_enabled=True,
        severe_anomaly_threshold=0.70,
    )


def test_base_priority_thresholds(agent):
    """Test standard RUL threshold mappings without anomaly."""
    # Critical: RUL <= 15
    d_crit = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=12.0, anomaly_score=0.1)
    assert d_crit.priority == "CRITICAL"
    assert d_crit.should_create_work_order is True

    # High: 15 < RUL <= 30
    d_high = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=25.0, anomaly_score=0.1)
    assert d_high.priority == "HIGH"
    assert d_high.should_create_work_order is True

    # Medium: 30 < RUL <= 60
    d_med = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=45.0, anomaly_score=0.1)
    assert d_med.priority == "MEDIUM"
    assert d_med.should_create_work_order is True

    # Low: RUL > 60
    d_low = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=85.0, anomaly_score=0.1)
    assert d_low.priority == "LOW"
    assert d_low.should_create_work_order is False


def test_anomaly_priority_escalation(agent):
    """Test that a severe anomaly raises priority level by 1 step."""
    # Base is LOW (RUL = 85), but severe anomaly (score = 0.85 >= 0.70)
    d_esc = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=85.0, anomaly_score=0.85)
    assert d_esc.priority == "MEDIUM"
    assert "ESCALATED" in d_esc.reason

    # Base is HIGH (RUL = 25), severe anomaly escalates to CRITICAL
    d_esc2 = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=25.0, anomaly_score=0.85)
    assert d_esc2.priority == "CRITICAL"

    # Base is CRITICAL: stays CRITICAL
    d_esc3 = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=10.0, anomaly_score=0.85)
    assert d_esc3.priority == "CRITICAL"


def test_missing_prediction_failsafe(agent):
    """
    CRITICAL DEFENSIVE GUARD:
    Missing or NaN prediction MUST NOT be assumed healthy!
    Must trigger fail-safe CRITICAL priority and ticket.
    """
    d_none = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=None)
    assert d_none.priority == "CRITICAL"
    assert d_none.should_create_work_order is True
    assert "Inference Exception" in d_none.reason

    d_nan = agent.evaluate_engine(engine_id=1, cycle=100, predicted_rul=float("nan"))
    assert d_nan.priority == "CRITICAL"
    assert d_nan.should_create_work_order is True


def test_invalid_data_failsafe(agent):
    """Invalid telemetry triggers immediate CRITICAL investigation."""
    d_bad = agent.evaluate_engine(
        engine_id=1, cycle=100, predicted_rul=50.0, data_valid=False, validation_error="Sensor harness disconnected"
    )
    assert d_bad.priority == "CRITICAL"
    assert "Data Quality Fault" in d_bad.reason
