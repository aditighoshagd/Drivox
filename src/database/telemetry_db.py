"""
Database and SQL Persistence Layer for Driver Safety Telemetry
Manages relational session storage, event logging, and fleet risk analytics using SQLite / SQL.
"""

import sqlite3
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class TelemetryDatabase:
    """
    Relational DBMS manager for driver fatigue sessions and safety telemetry events.
    """

    def __init__(self, db_path: str = "driver_safety.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_schema(self):
        """Creates normalized SQL schema for sessions and telemetry event logging."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table 1: Driver Sessions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS driver_sessions (
                session_id TEXT PRIMARY KEY,
                driver_id TEXT NOT NULL,
                vehicle_id TEXT DEFAULT 'FLEET-001',
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL DEFAULT 0.0,
                total_frames INTEGER DEFAULT 0,
                total_alerts INTEGER DEFAULT 0,
                avg_ear REAL DEFAULT 0.0,
                avg_mar REAL DEFAULT 0.0,
                max_fatigue_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'ACTIVE'
            );
            """)

            # Table 2: Safety Telemetry Events
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                ear_value REAL,
                mar_value REAL,
                fatigue_score REAL,
                latency_ms REAL,
                head_pose_json TEXT,
                FOREIGN KEY (session_id) REFERENCES driver_sessions (session_id)
            );
            """)

            # Indexes for high-throughput queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON safety_events (session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON safety_events (timestamp);")
            conn.commit()

    def create_session(self, session_id: str, driver_id: str = "DRIVER_DEFAULT", vehicle_id: str = "VEHICLE_01"):
        """Inserts a new driver session record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO driver_sessions (session_id, driver_id, vehicle_id, start_time, status)
                VALUES (?, ?, ?, ?, 'ACTIVE')
            """, (session_id, driver_id, vehicle_id, datetime.utcnow().isoformat()))
            conn.commit()

    def log_safety_event(
        self,
        session_id: str,
        event_type: str,
        ear: float,
        mar: float,
        fatigue_score: float,
        latency_ms: float,
        head_pose: Optional[Dict[str, float]] = None
    ):
        """Logs a single critical fatigue or distraction event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO safety_events (session_id, timestamp, event_type, ear_value, mar_value, fatigue_score, latency_ms, head_pose_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.utcnow().isoformat(),
                event_type,
                ear,
                mar,
                fatigue_score,
                latency_ms,
                json.dumps(head_pose or {})
            ))
            # Increment session alert count
            cursor.execute("""
                UPDATE driver_sessions
                SET total_alerts = total_alerts + 1
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()

    def end_session(
        self,
        session_id: str,
        total_frames: int,
        avg_ear: float,
        avg_mar: float,
        max_fatigue_score: float
    ):
        """Finalizes an active session with aggregated telemetry metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Calculate duration
            cursor.execute("SELECT start_time FROM driver_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            duration = 0.0
            if row:
                start_dt = datetime.fromisoformat(row["start_time"])
                duration = (datetime.utcnow() - start_dt).total_seconds()

            cursor.execute("""
                UPDATE driver_sessions
                SET end_time = ?,
                    duration_seconds = ?,
                    total_frames = ?,
                    avg_ear = ?,
                    avg_mar = ?,
                    max_fatigue_score = ?,
                    status = 'COMPLETED'
                WHERE session_id = ?
            """, (
                datetime.utcnow().isoformat(),
                duration,
                total_frames,
                avg_ear,
                avg_mar,
                max_fatigue_score,
                session_id
            ))
            conn.commit()

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves summary report for a specific driving session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM driver_sessions WHERE session_id = ?", (session_id,))
            session = cursor.fetchone()
            if not session:
                return None

            cursor.execute("""
                SELECT event_type, COUNT(*) as count
                FROM safety_events
                WHERE session_id = ?
                GROUP BY event_type
            """, (session_id,))
            events = {row["event_type"]: row["count"] for row in cursor.fetchall()}

            return {
                "session_id": session["session_id"],
                "driver_id": session["driver_id"],
                "vehicle_id": session["vehicle_id"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "duration_minutes": round(session["duration_seconds"] / 60.0, 2),
                "total_frames": session["total_frames"],
                "total_alerts": session["total_alerts"],
                "avg_ear": round(session["avg_ear"], 4),
                "avg_mar": round(session["avg_mar"], 4),
                "max_fatigue_score": round(session["max_fatigue_score"], 3),
                "events_breakdown": events
            }

    def get_fleet_risk_analytics(self) -> List[Dict[str, Any]]:
        """Executes aggregate SQL query across fleet sessions to assess high-risk drivers."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    driver_id,
                    COUNT(session_id) as total_trips,
                    SUM(total_alerts) as cumulative_alerts,
                    AVG(avg_ear) as fleet_avg_ear,
                    MAX(max_fatigue_score) as highest_fatigue_observed
                FROM driver_sessions
                GROUP BY driver_id
                ORDER BY cumulative_alerts DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
