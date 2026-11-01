"""
Unit tests for SQLite persistent work order service.
Tests duplicate ticket suppression, idempotency, lifecycle status transitions, and CSV exports.
"""

import os
import pytest
import tempfile
from src.database import init_db
from src.work_order_service import WorkOrderService
from src.triage_agent import RuleBasedTriageAgent

import os
import sys

# Adds the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def temp_wo_service(tmp_path):
    """Create a temporary SQLite database for clean test isolation."""
    db_path = str(tmp_path / "test_work_orders.db")
    service = WorkOrderService(db_path=db_path)
    return service



def test_work_order_creation_and_persistence(temp_wo_service):
    """Test standard work order creation and database retrieval."""
    agent = RuleBasedTriageAgent()
    decision = agent.evaluate_engine(engine_id=42, cycle=150, predicted_rul=10.0)

    wo = temp_wo_service.create_work_order_from_decision(decision)
    assert wo is not None
    assert wo["work_order_id"] == decision.work_order_id
    assert wo["engine_id"] == 42
    assert wo["priority"] == "CRITICAL"
    assert wo["status"] == "OPEN"

    # Query back from DB
    retrieved = temp_wo_service.get_work_order(decision.work_order_id)
    assert retrieved["work_order_id"] == decision.work_order_id


def test_duplicate_work_order_prevention(temp_wo_service):
    """
    CRITICAL IDEMPOTENCY TEST:
    Subsequent cycles for an engine that already has an active OPEN work order
    MUST NOT generate redundant tickets.
    """
    agent = RuleBasedTriageAgent()
    dec1 = agent.evaluate_engine(engine_id=42, cycle=150, predicted_rul=10.0)
    wo1 = temp_wo_service.create_work_order_from_decision(dec1)
    assert wo1 is not None

    # Next cycle, still critical
    dec2 = agent.evaluate_engine(engine_id=42, cycle=151, predicted_rul=9.0)
    wo2 = temp_wo_service.create_work_order_from_decision(dec2)
    assert wo2 is None, "Duplicate active work order was created instead of being suppressed!"

    # Total work orders in database must still be exactly 1
    wos = temp_wo_service.list_work_orders(engine_id_filter=42)
    assert len(wos) == 1


def test_status_lifecycle_transitions(temp_wo_service):
    """Test transition from OPEN -> IN_PROGRESS -> COMPLETED."""
    agent = RuleBasedTriageAgent()
    decision = agent.evaluate_engine(engine_id=7, cycle=80, predicted_rul=14.0)
    wo = temp_wo_service.create_work_order_from_decision(decision)
    wo_id = wo["work_order_id"]

    # Transition to IN_PROGRESS
    ok = temp_wo_service.update_work_order_status(wo_id, "IN_PROGRESS", note="Technician assigned")
    assert ok is True
    assert temp_wo_service.get_work_order(wo_id)["status"] == "IN_PROGRESS"

    # Transition to COMPLETED
    ok2 = temp_wo_service.update_work_order_status(wo_id, "COMPLETED", note="Inspection passed")
    assert ok2 is True
    completed_wo = temp_wo_service.get_work_order(wo_id)
    assert completed_wo["status"] == "COMPLETED"
    assert completed_wo["closed_at"] is not None


def test_export_to_csv(temp_wo_service, tmp_path):
    """Test exporting work orders to CSV."""
    agent = RuleBasedTriageAgent()
    dec = agent.evaluate_engine(engine_id=1, cycle=20, predicted_rul=5.0)
    temp_wo_service.create_work_order_from_decision(dec)

    export_path = str(tmp_path / "exported_wos.csv")
    temp_wo_service.export_to_csv(export_path)
    assert os.path.exists(export_path)

    import pandas as pd
    df_exp = pd.read_csv(export_path)
    assert len(df_exp) == 1
    assert df_exp["engine_id"].iloc[0] == 1
