"""
Work order management service backed by SQLite.
Handles persistent lifecycle tracking of maintenance work orders with idempotency checks,
duplicate prevention, status transitions (OPEN -> IN_PROGRESS -> COMPLETED), and CSV reporting.
"""

import os
import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import sqlite3
import logging

import os
import sys

# Adds the project root directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import get_db_connection, init_db, DEFAULT_DB_PATH
from src.triage_agent import TriageDecision

logger = logging.getLogger(__name__)

VALID_STATUSES = ["OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"]


class WorkOrderService:
    """
    Enterprise-grade maintenance work order service with duplicate prevention,
    lifecycle audits, and automated triage integration.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def create_work_order_from_decision(
        self,
        decision: TriageDecision,
        assignee: str = "Fleet Reliability Engineering",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new maintenance work order if eligible and not a duplicate.
        Ensures idempotent operation: if an OPEN or IN_PROGRESS work order
        already exists for this engine, prevents creating redundant tickets.
        """
        if not decision.should_create_work_order:
            logger.debug(f"Skipping work order for Engine {decision.engine_id}: Priority {decision.priority} below threshold.")
            return None

        conn = get_db_connection(self.db_path)
        with conn:
            # Check for existing active work order for this engine
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT work_order_id, cycle, priority, status FROM work_orders
                WHERE engine_id = ? AND status IN ('OPEN', 'IN_PROGRESS');
                """,
                (decision.engine_id,),
            )
            active_wo = cursor.fetchone()

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            if active_wo:
                existing_id = active_wo["work_order_id"]
                existing_prio = active_wo["priority"]
                logger.info(
                    f"Idempotent guard: Engine {decision.engine_id} already has active work order "
                    f"'{existing_id}' (Priority: {existing_prio}). Duplicate creation prevented."
                )
                # Audit the suppressed duplicate
                conn.execute(
                    """
                    INSERT INTO audit_log (event_type, entity_id, description, created_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (
                        "DUPLICATE_WORK_ORDER_SUPPRESSED",
                        existing_id,
                        f"Engine {decision.engine_id} triggered {decision.priority} at cycle {decision.cycle}, suppressed redundant creation.",
                        now_iso,
                    ),
                )
                return None

            # Insert new work order using INSERT OR IGNORE to prevent unique constraint crashes
            predicted_rul_val = decision.predicted_rul if decision.predicted_rul is not None else -1.0

            conn.execute(
                """
                INSERT OR IGNORE INTO work_orders (
                    work_order_id, engine_id, cycle, predicted_rul, anomaly_score,
                    anomaly_status, priority, status, reason, recommended_action,
                    assignee, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?);
                """,
                (
                    decision.work_order_id,
                    decision.engine_id,
                    decision.cycle,
                    predicted_rul_val,
                    decision.anomaly_score,
                    decision.anomaly_status,
                    decision.priority,
                    decision.reason,
                    decision.recommended_action,
                    assignee,
                    now_iso,
                    now_iso,
                ),
            )

            # Record in audit log
            conn.execute(
                """
                INSERT INTO audit_log (event_type, entity_id, description, created_at)
                VALUES (?, ?, ?, ?);
                """,
                (
                    "WORK_ORDER_CREATED",
                    decision.work_order_id,
                    f"Created {decision.priority} work order for Engine {decision.engine_id} at cycle {decision.cycle}",
                    now_iso,
                ),
            )

        logger.info(f"Dispatched Work Order {decision.work_order_id} for Engine {decision.engine_id}")
        return self.get_work_order(decision.work_order_id)

    def update_work_order_status(
        self,
        work_order_id: str,
        new_status: str,
        note: Optional[str] = None,
    ) -> bool:
        """
        Transition work order lifecycle status (OPEN -> IN_PROGRESS -> COMPLETED).
        """
        new_status = new_status.upper()
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}")

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        closed_at = now_iso if new_status in ["COMPLETED", "CANCELLED"] else None

        conn = get_db_connection(self.db_path)
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE work_orders
                SET status = ?, updated_at = ?, closed_at = COALESCE(?, closed_at)
                WHERE work_order_id = ?;
                """,
                (new_status, now_iso, closed_at, work_order_id),
            )

            if cursor.rowcount == 0:
                logger.warning(f"Work order {work_order_id} not found.")
                return False

            audit_desc = f"Status updated to {new_status}" + (f": {note}" if note else "")
            conn.execute(
                """
                INSERT INTO audit_log (event_type, entity_id, description, created_at)
                VALUES (?, ?, ?, ?);
                """,
                ("WORK_ORDER_UPDATED", work_order_id, audit_desc, now_iso),
            )

        logger.info(f"Updated work order {work_order_id} to {new_status}")
        return True

    def get_work_order(self, work_order_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single work order by identifier."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM work_orders WHERE work_order_id = ?;", (work_order_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_work_orders(
        self,
        status_filter: Optional[str] = None,
        priority_filter: Optional[str] = None,
        engine_id_filter: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query work orders with optional filtering."""
        query = "SELECT * FROM work_orders WHERE 1=1"
        params: List[Any] = []

        if status_filter:
            query += " AND status = ?"
            params.append(status_filter.upper())
        if priority_filter:
            query += " AND priority = ?"
            params.append(priority_filter.upper())
        if engine_id_filter is not None:
            query += " AND engine_id = ?"
            params.append(int(engine_id_filter))

        query += " ORDER BY created_at DESC;"

        conn = get_db_connection(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Aggregate high-level metrics for dashboard cards."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM work_orders;")
        total_wo = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM work_orders WHERE status = 'OPEN';")
        open_wo = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM work_orders WHERE priority = 'CRITICAL' AND status != 'COMPLETED';")
        critical_active = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM work_orders WHERE priority = 'HIGH' AND status != 'COMPLETED';")
        high_active = cursor.fetchone()[0]

        return {
            "total_work_orders": total_wo,
            "open_work_orders": open_wo,
            "critical_active_alerts": critical_active,
            "high_active_alerts": high_active,
        }

    def export_to_csv(self, filepath: str) -> str:
        """Export all historical work orders to CSV."""
        df = self.list_work_orders()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} work orders to {filepath}")
        return filepath