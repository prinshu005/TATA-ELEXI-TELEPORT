"""
Database management module for predictive maintenance platform.
Manages persistent SQLite storage for maintenance work orders, telemetry snapshots,
and MLOps audit logs with WAL mode enabled for high-concurrency Streamlit dashboards.
"""

import os
import sqlite3
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "database/work_orders.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with row factory and WAL mode."""
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables, indexes, and initial configuration."""
    conn = get_db_connection(db_path)
    with conn:
        # 1. Maintenance Work Orders
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS work_orders (
                work_order_id TEXT PRIMARY KEY,
                engine_id INTEGER NOT NULL,
                cycle INTEGER NOT NULL,
                predicted_rul REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_status TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                reason TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                assignee TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wo_engine_status
            ON work_orders (engine_id, status);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_wo_priority
            ON work_orders (priority);
            """
        )

        # 2. Fleet Telemetry Snapshots
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fleet_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                engine_id INTEGER NOT NULL,
                cycle INTEGER NOT NULL,
                predicted_rul REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                anomaly_status TEXT NOT NULL,
                priority TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_telemetry_engine_cycle
            ON fleet_telemetry (engine_id, cycle);
            """
        )

        # 3. System Audit Log
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_id TEXT,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    logger.info(f"Database initialized at {db_path}")
