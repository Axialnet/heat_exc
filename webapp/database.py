"""
Database persistence layer for CHO Heat Exchanger Monitoring Platform.
Uses aiosqlite for async SQLite database operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from webapp.config import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- 1. Raw sensor readings as ingested from Version 1 source
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchanger_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    time_h REAL NOT NULL,
    timestamp_utc TEXT NOT NULL,
    T_in_measured_C REAL,
    m_dot_measured_kg_s REAL,
    T_in_C REAL,
    T_in_K REAL,
    m_dot_nominal_kg_s REAL,
    Re REAL,
    u_m_s REAL,
    tau_w_Pa REAL,
    dP_Pa REAL,
    Rf_m2K_W REAL,
    U_clean_W_m2K REAL,
    U_overall_W_m2K REAL,
    Q_W REAL,
    Q_clean_W REAL,
    thermal_efficiency REAL,
    fouling_factor_TEMA TEXT,
    cip_event INTEGER,
    cip_effectiveness REAL,
    R_wall_m2K_W REAL,
    U_total_W_m2K REAL,
    Q_total_W REAL,
    efficiency_total REAL,
    degradation_source TEXT,
    rho_kg_m3 REAL,
    mu_Pa_s REAL,
    created_at TEXT NOT NULL
);

-- Index for efficient time-series queries
CREATE INDEX IF NOT EXISTS idx_sensor_time ON sensor_readings (exchanger_id, time_h);
CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings (exchanger_id, timestamp_utc);

-- 2. Derived physical indicators, model predictions, and health/risk metrics
CREATE TABLE IF NOT EXISTS derived_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER,
    exchanger_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    time_h REAL NOT NULL,
    timestamp_utc TEXT NOT NULL,
    di_fouling_actual REAL,
    di_fouling_predicted REAL,
    di_total REAL,
    u_retention REAL,
    q_retention REAL,
    thermal_efficiency REAL,
    performance_score REAL,
    fouling_risk REAL,
    maintenance_risk REAL,
    maintenance_state TEXT,
    degradation_rate REAL,
    ttt_warning_h REAL,
    ttt_critical_h REAL,
    recommendation_state TEXT,
    recommendation_text TEXT,
    consecutive_above_warn INTEGER,
    consecutive_above_crit INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (reading_id) REFERENCES sensor_readings(id)
);

CREATE INDEX IF NOT EXISTS idx_derived_time ON derived_metrics (exchanger_id, time_h);
CREATE INDEX IF NOT EXISTS idx_derived_timestamp ON derived_metrics (exchanger_id, timestamp_utc);

-- 3. Alert event records with deduplication tracking
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    exchanger_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    severity TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    di_current REAL,
    threshold REAL,
    ttt_h REAL,
    risk_score REAL,
    recommended_action TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_alerts_exchanger ON alerts (exchanger_id, severity);

-- 4. Clean-in-Place (CIP) operational events
CREATE TABLE IF NOT EXISTS cip_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchanger_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    time_h REAL NOT NULL,
    timestamp_utc TEXT NOT NULL,
    event_type TEXT NOT NULL,
    cip_effectiveness REAL,
    di_before REAL,
    di_after REAL,
    details TEXT,
    created_at TEXT NOT NULL
);

-- 5. Persistent ingestion state and replay cursor
CREATE TABLE IF NOT EXISTS ingestion_state (
    exchanger_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL,
    last_row_index INTEGER NOT NULL,
    last_time_h REAL NOT NULL,
    records_processed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    updated_at TEXT NOT NULL
);
"""


class Database:
    """Async database interface for CHO monitoring."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.database_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize schema tables and indices."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
            logger.info("Database schema initialized at %s", self.db_path)

    async def get_connection(self) -> aiosqlite.Connection:
        """Get an active aiosqlite connection with row factory enabled."""
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    # ── Ingestion State Management ──────────────────────────────────────

    async def get_ingestion_state(self, exchanger_id: str) -> Optional[Dict[str, Any]]:
        """Get the last saved replay cursor and stats."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ingestion_state WHERE exchanger_id = ?",
                (exchanger_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def save_ingestion_state(
        self,
        exchanger_id: str,
        scenario_id: str,
        row_index: int,
        time_h: float,
        records_processed: int,
        status: str = "RUNNING"
    ) -> None:
        """Upsert current replay cursor state."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO ingestion_state (
                    exchanger_id, scenario_id, last_row_index, last_time_h,
                    records_processed, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchanger_id) DO UPDATE SET
                    scenario_id = excluded.scenario_id,
                    last_row_index = excluded.last_row_index,
                    last_time_h = excluded.last_time_h,
                    records_processed = excluded.records_processed,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (exchanger_id, scenario_id, row_index, time_h, records_processed, status, now)
            )
            await db.commit()

    # ── Record Insertion ───────────────────────────────────────────────

    async def insert_cycle_data(
        self,
        raw_reading: Dict[str, Any],
        derived: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        cip_event: Optional[Dict[str, Any]] = None
    ) -> int:
        """Atomically insert raw reading, derived metrics, alerts, and optional CIP event."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            # 1. Raw sensor reading
            cursor = await db.execute(
                """
                INSERT INTO sensor_readings (
                    exchanger_id, scenario_id, time_h, timestamp_utc,
                    T_in_measured_C, m_dot_measured_kg_s, T_in_C, T_in_K,
                    m_dot_nominal_kg_s, Re, u_m_s, tau_w_Pa, dP_Pa,
                    Rf_m2K_W, U_clean_W_m2K, U_overall_W_m2K, Q_W, Q_clean_W,
                    thermal_efficiency, fouling_factor_TEMA, cip_event,
                    cip_effectiveness, R_wall_m2K_W, U_total_W_m2K, Q_total_W,
                    efficiency_total, degradation_source, rho_kg_m3, mu_Pa_s, created_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    raw_reading.get("exchanger_id", settings.scenario_id),
                    raw_reading.get("scenario_id", settings.scenario_id),
                    raw_reading.get("time_h", 0.0),
                    raw_reading.get("timestamp_utc", now),
                    raw_reading.get("T_in_measured_C"),
                    raw_reading.get("m_dot_measured_kg_s"),
                    raw_reading.get("T_in_C"),
                    raw_reading.get("T_in_K"),
                    raw_reading.get("m_dot_nominal_kg_s"),
                    raw_reading.get("Re"),
                    raw_reading.get("u_m_s"),
                    raw_reading.get("tau_w_Pa"),
                    raw_reading.get("dP_Pa"),
                    raw_reading.get("Rf_m2K_W"),
                    raw_reading.get("U_clean_W_m2K"),
                    raw_reading.get("U_overall_W_m2K"),
                    raw_reading.get("Q_W"),
                    raw_reading.get("Q_clean_W"),
                    raw_reading.get("thermal_efficiency"),
                    str(raw_reading.get("fouling_factor_TEMA", "")),
                    int(raw_reading.get("cip_event", 0)),
                    raw_reading.get("cip_effectiveness", 0.0),
                    raw_reading.get("R_wall_m2K_W"),
                    raw_reading.get("U_total_W_m2K"),
                    raw_reading.get("Q_total_W"),
                    raw_reading.get("efficiency_total"),
                    str(raw_reading.get("degradation_source", "")),
                    raw_reading.get("rho_kg_m3"),
                    raw_reading.get("mu_Pa_s"),
                    now
                )
            )
            reading_id = cursor.lastrowid

            # 2. Derived metrics
            await db.execute(
                """
                INSERT INTO derived_metrics (
                    reading_id, exchanger_id, scenario_id, time_h, timestamp_utc,
                    di_fouling_actual, di_fouling_predicted, di_total,
                    u_retention, q_retention, thermal_efficiency,
                    performance_score, fouling_risk, maintenance_risk,
                    maintenance_state, degradation_rate, ttt_warning_h,
                    ttt_critical_h, recommendation_state, recommendation_text,
                    consecutive_above_warn, consecutive_above_crit, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reading_id,
                    derived.get("exchanger_id", settings.scenario_id),
                    derived.get("scenario_id", settings.scenario_id),
                    derived.get("time_h", 0.0),
                    derived.get("timestamp_utc", now),
                    derived.get("di_fouling_actual"),
                    derived.get("di_fouling_predicted"),
                    derived.get("di_total"),
                    derived.get("u_retention"),
                    derived.get("q_retention"),
                    derived.get("thermal_efficiency"),
                    derived.get("performance_score"),
                    derived.get("fouling_risk"),
                    derived.get("maintenance_risk"),
                    derived.get("maintenance_state"),
                    derived.get("degradation_rate"),
                    derived.get("ttt_warning_h"),
                    derived.get("ttt_critical_h"),
                    derived.get("recommendation_state"),
                    derived.get("recommendation_text"),
                    derived.get("consecutive_above_warn", 0),
                    derived.get("consecutive_above_crit", 0),
                    now
                )
            )

            # 3. Alerts
            for alt in alerts:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO alerts (
                        id, exchanger_id, scenario_id, timestamp_utc,
                        severity, trigger_reason, di_current, threshold,
                        ttt_h, risk_score, recommended_action, acknowledged, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alt["id"],
                        alt["exchanger_id"],
                        alt.get("scenario_id", settings.scenario_id),
                        alt["timestamp"],
                        alt["severity"],
                        alt["reason"],
                        alt["current_DI"],
                        alt["threshold"],
                        alt.get("ttt"),
                        alt["risk_score"],
                        alt["recommended_action"],
                        0,
                        now
                    )
                )

            # 4. Optional CIP event
            if cip_event:
                await db.execute(
                    """
                    INSERT INTO cip_events (
                        exchanger_id, scenario_id, time_h, timestamp_utc,
                        event_type, cip_effectiveness, di_before, di_after, details, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cip_event.get("exchanger_id", settings.scenario_id),
                        cip_event.get("scenario_id", settings.scenario_id),
                        cip_event.get("time_h", 0.0),
                        cip_event.get("timestamp_utc", now),
                        cip_event.get("event_type", "DETECTED"),
                        cip_event.get("cip_effectiveness", 0.0),
                        cip_event.get("di_before"),
                        cip_event.get("di_after"),
                        cip_event.get("details", ""),
                        now
                    )
                )

            await db.commit()
            return reading_id

    # ── Queries for API ────────────────────────────────────────────────

    async def get_latest_state(self, exchanger_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent sensor reading joined with its derived metrics."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = """
            SELECT
                s.exchanger_id, s.scenario_id, s.time_h, s.timestamp_utc,
                s.T_in_measured_C, s.m_dot_measured_kg_s, s.Re, s.u_m_s,
                s.tau_w_Pa, s.dP_Pa, s.Rf_m2K_W, s.U_clean_W_m2K,
                s.U_overall_W_m2K, s.Q_W, s.Q_clean_W, s.thermal_efficiency,
                s.rho_kg_m3, s.mu_Pa_s, s.cip_event, s.cip_effectiveness,
                d.di_fouling_actual, d.di_fouling_predicted, d.di_total,
                d.u_retention, d.q_retention, d.performance_score,
                d.fouling_risk, d.maintenance_risk, d.maintenance_state,
                d.degradation_rate, d.ttt_warning_h, d.ttt_critical_h,
                d.recommendation_state, d.recommendation_text,
                d.consecutive_above_warn, d.consecutive_above_crit
            FROM sensor_readings s
            JOIN derived_metrics d ON s.id = d.reading_id
            WHERE s.exchanger_id = ?
            ORDER BY s.id DESC
            LIMIT 1
            """
            async with db.execute(query, (exchanger_id,)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_history(
        self,
        exchanger_id: str,
        hours: float = 168.0,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Fetch historical readings for time-series charts (downsampled if large)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # First get max time_h to determine window
            async with db.execute(
                "SELECT MAX(time_h) as max_t FROM derived_metrics WHERE exchanger_id = ?",
                (exchanger_id,)
            ) as cur:
                r = await cur.fetchone()
                max_t = r["max_t"] if r and r["max_t"] is not None else 0.0

            min_t = max(0.0, max_t - hours)
            query = """
            SELECT
                d.time_h, d.timestamp_utc,
                d.di_fouling_actual, d.di_fouling_predicted,
                d.performance_score, d.maintenance_risk, d.maintenance_state,
                d.u_retention, d.q_retention, d.degradation_rate,
                d.ttt_critical_h,
                s.T_in_measured_C, s.m_dot_measured_kg_s, s.dP_Pa, s.Rf_m2K_W
            FROM derived_metrics d
            JOIN sensor_readings s ON d.reading_id = s.id
            WHERE d.exchanger_id = ? AND d.time_h >= ?
            ORDER BY d.time_h ASC
            LIMIT ?
            """
            async with db.execute(query, (exchanger_id, min_t, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_alerts(
        self,
        exchanger_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve recent alert events."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            clauses = []
            params = []
            if exchanger_id:
                clauses.append("exchanger_id = ?")
                params.append(exchanger_id)
            if severity and severity != "ALL":
                clauses.append("severity = ?")
                params.append(severity)

            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            query = f"SELECT * FROM alerts {where} ORDER BY timestamp_utc DESC LIMIT ?"
            params.append(limit)

            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_cip_events(self, exchanger_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent CIP events."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = """
            SELECT * FROM cip_events
            WHERE exchanger_id = ?
            ORDER BY time_h DESC
            LIMIT ?
            """
            async with db.execute(query, (exchanger_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def get_database_stats(self) -> Dict[str, Any]:
        """Summary stats for system diagnostics."""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}
            for table in ["sensor_readings", "derived_metrics", "alerts", "cip_events"]:
                async with db.execute(f"SELECT COUNT(*) FROM {table}") as cur:
                    r = await cur.fetchone()
                    stats[f"{table}_count"] = r[0] if r else 0
            return stats
