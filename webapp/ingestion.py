"""
Continuous Data Ingestion and Processing Service for CHO Heat Exchanger Monitoring.
Reads Version 1 data source in real-time or replay mode, runs physical calculations,
model inference, health/risk engines, alert generation, and WebSocket broadcasting.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from starlette.websockets import WebSocket, WebSocketState

from webapp.config import settings
from webapp.database import Database
from webapp.engine import (
    AlertEngine,
    CleaningRecommendation,
    FeatureEngineering,
    HealthScoring,
    ModelInference,
    PhysicalCalculations,
    RiskEngine,
    WaterProperties,
)

logger = logging.getLogger(__name__)


class IngestionWorker:
    """
    24/7 autonomous background worker. Replays or ingests records from the
    Version 1 source, driving the entire physical-calculation and ML decision pipeline.
    """

    def __init__(self, db: Database):
        self.db = db
        self.fe = FeatureEngineering()
        self.model = ModelInference()
        self.risk_engine = RiskEngine()
        self.alert_engine = AlertEngine()
        self.cleaning_engine = CleaningRecommendation()

        # Operational state
        self.scenario_id = settings.scenario_id
        self.exchanger_id = settings.scenario_id
        self.is_running = False
        self.is_paused = False
        self.replay_interval = settings.replay_interval_seconds

        # Internal tracking
        self.cursor_index = 0
        self.records_processed = 0
        self.last_reading_time_utc: Optional[str] = None
        self.last_time_h: float = 0.0
        self.last_state_packet: Optional[Dict[str, Any]] = None
        self.error_count = 0
        self.last_error: Optional[str] = None

        # Data cache for V1 replay
        self._df: Optional[pd.DataFrame] = None
        self._total_rows = 0

        # WebSocket subscribers
        self._active_connections: Set[WebSocket] = set()

    # ── WebSocket Management ────────────────────────────────────────────

    async def connect_client(self, websocket: WebSocket) -> None:
        """Register a new active WebSocket connection."""
        await websocket.accept()
        self._active_connections.add(websocket)
        logger.info("WebSocket client connected. Total clients: %d", len(self._active_connections))
        # Immediately push the latest state if available
        if self.last_state_packet:
            try:
                await websocket.send_text(json.dumps(self.last_state_packet))
            except Exception:
                pass

    def disconnect_client(self, websocket: WebSocket) -> None:
        """Remove a closed WebSocket connection."""
        self._active_connections.discard(websocket)
        logger.info("WebSocket client disconnected. Total clients: %d", len(self._active_connections))

    async def broadcast_state(self, packet: Dict[str, Any]) -> None:
        """Broadcast live telemetry to all connected clients."""
        if not self._active_connections:
            return
        message = json.dumps(packet)
        disconnected = set()
        for ws in self._active_connections:
            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_text(message)
                except Exception:
                    disconnected.add(ws)
            else:
                disconnected.add(ws)
        for ws in disconnected:
            self._active_connections.discard(ws)

    # ── Dataset Loading ─────────────────────────────────────────────────

    def load_dataset(self) -> None:
        """Load and filter the V1 dataset to the active scenario, utilizing fast cache when available."""
        cache_dir = Path(settings.v1_data_path).parent / "cache"
        cached_file = cache_dir / f"{self.scenario_id}.csv"

        if cached_file.exists():
            logger.info("Loading cached scenario from %s...", cached_file)
            scenario_df = pd.read_csv(cached_file)
        else:
            v1_path = Path(settings.v1_data_path)
            if not v1_path.exists():
                raise FileNotFoundError(f"V1 dataset not found at {v1_path}")

            logger.info("Loading full V1 dataset from %s for scenario %s...", v1_path, self.scenario_id)
            df_all = pd.read_csv(v1_path)
            scenario_df = df_all[df_all["scenario_id"] == self.scenario_id].sort_values("time_h").reset_index(drop=True)
            if scenario_df.empty:
                available = df_all["scenario_id"].unique()
                self.scenario_id = str(available[0])
                self.exchanger_id = self.scenario_id
                scenario_df = df_all[df_all["scenario_id"] == self.scenario_id].sort_values("time_h").reset_index(drop=True)
                logger.warning("Configured scenario missing. Defaulted to %s", self.scenario_id)

            # Auto-populate cache for subsequent instant boots
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                scenario_df.to_csv(cached_file, index=False)
            except Exception as e:
                logger.warning("Could not cache scenario: %s", e)

        self._df = scenario_df
        self._total_rows = len(scenario_df)
        logger.info("Loaded %d rows for scenario %s (time_h: %s to %s)",
                    self._total_rows, self.scenario_id,
                    scenario_df["time_h"].min(), scenario_df["time_h"].max())

    # ── Resume / State Restoration ──────────────────────────────────────

    async def restore_cursor(self) -> None:
        """Restore replay cursor from persistent ingestion_state table."""
        state = await self.db.get_ingestion_state(self.exchanger_id)
        if state:
            self.cursor_index = int(state.get("last_row_index", 0))
            self.records_processed = int(state.get("records_processed", 0))
            self.last_time_h = float(state.get("last_time_h", 0.0))
            if self._total_rows > 0 and self.cursor_index >= self._total_rows:
                if settings.replay_loop:
                    self.cursor_index = 0
                    logger.info("Restored cursor was at end of dataset (%d rows). Resetting to 0 for infinite loop.", self._total_rows)
            logger.info("Resumed ingestion at row %d (time_h: %.1f) from database",
                        self.cursor_index, self.last_time_h)
        else:
            self.cursor_index = 0
            self.records_processed = 0
            logger.info("Starting fresh ingestion from beginning of scenario")

    # ── Single Record Processing Pipeline ────────────────────────────────

    async def process_next_record(self) -> Optional[Dict[str, Any]]:
        """
        Execute the complete 1-minute cycle:
        1. Fetch next record from source
        2. Calculate physical properties (rho, mu)
        3. Feature engineering
        4. Model inference
        5. Physical indicators & health score
        6. Risk & TTT estimation
        7. Alert evaluation (deduplicated)
        8. Cleaning recommendation (persistence filtered)
        9. Persist to SQLite
        10. Return state packet
        """
        if self._df is None or self._total_rows == 0:
            return None

        if self.cursor_index >= self._total_rows:
            if settings.replay_loop:
                logger.info("Reached end of scenario dataset (%d rows). Looping back to start.", self._total_rows)
                self.cursor_index = 0
                self.fe = FeatureEngineering()  # reset feature state on loop
                self.risk_engine.reset_history()  # clean slope calculation window
            else:
                logger.info("Reached end of scenario dataset. Replay complete.")
                await self.db.save_ingestion_state(
                    self.exchanger_id, self.scenario_id, self.cursor_index,
                    self.last_time_h, self.records_processed, status="COMPLETED"
                )
                self.is_running = False
                return None

        # Fetch current record as dictionary
        row = self._df.iloc[self.cursor_index].to_dict()
        now_utc = datetime.now(timezone.utc).isoformat()
        time_h = float(row.get("time_h", 0.0))

        try:
            # 1. Compute missing fluid thermophysical properties for water
            T_meas = float(row["T_in_measured_C"])
            rho_calc = WaterProperties.density(T_meas)
            mu_calc = WaterProperties.viscosity(T_meas)

            row["rho_kg_m3"] = rho_calc
            row["mu_Pa_s"] = mu_calc
            row["exchanger_id"] = self.exchanger_id
            row["timestamp_utc"] = now_utc

            # 2. Run feature engineering (16 strictly causal CIP-aware features)
            features = self.fe.process_record(row)

            # 3. Model inference (Track A production model)
            di_predicted = self.model.predict(features)

            # 4. Physical performance metrics
            U_clean = float(row.get("U_clean_W_m2K", 1.0))
            U_overall = float(row.get("U_overall_W_m2K", 1.0))
            U_total = float(row.get("U_total_W_m2K", U_overall))
            Q_W = float(row.get("Q_W", 0.0))
            Q_clean = float(row.get("Q_clean_W", Q_W))

            di_actual = PhysicalCalculations.di_fouling(U_overall, U_clean)
            di_total = PhysicalCalculations.di_total(U_total, U_clean)
            u_ret = PhysicalCalculations.u_retention(U_overall, U_clean)
            q_ret = PhysicalCalculations.q_retention(Q_W, Q_clean)
            th_eff = float(row.get("thermal_efficiency", 1.0))

            # 5. Composite Performance / Health Score (Phase 8: 0-100)
            perf_score = HealthScoring.performance_score(u_ret, q_ret, di_actual)

            # 6. Risk Engine & TTT Projection (Phase 9)
            # Evaluate using actual DI (or predicted if actual unavailable in production)
            eval_di = di_actual if di_actual is not None else (di_predicted or 0.0)
            risk_res = self.risk_engine.evaluate(time_h, eval_di, perf_score)

            # 7. Alert System (Phase 10: 24h deduplicated)
            raw_alerts = self.alert_engine.evaluate(
                exchanger_id=self.exchanger_id,
                DI=eval_di,
                ttt_critical=risk_res.ttt_critical_h,
                risk_score=risk_res.maintenance_risk
            )
            alerts_list = [
                {
                    "id": a.id,
                    "exchanger_id": a.exchanger_id,
                    "scenario_id": self.scenario_id,
                    "timestamp": a.timestamp,
                    "severity": a.severity,
                    "reason": a.reason,
                    "current_DI": a.current_DI,
                    "threshold": a.threshold,
                    "ttt": a.ttt,
                    "risk_score": a.risk_score,
                    "recommended_action": a.recommended_action
                }
                for a in raw_alerts
            ]

            # 8. Cleaning Recommendation Engine (Phase 11: persistence filtered)
            clean_res = self.cleaning_engine.evaluate(
                DI=eval_di,
                ttt_critical=risk_res.ttt_critical_h,
                risk_score=risk_res.maintenance_risk
            )

            # 9. CIP event detection
            cip_record = None
            if int(row.get("cip_event", 0)) == 1 or int(round(time_h)) in settings.cip_hours:
                cip_record = {
                    "exchanger_id": self.exchanger_id,
                    "scenario_id": self.scenario_id,
                    "time_h": time_h,
                    "timestamp_utc": now_utc,
                    "event_type": "SCHEDULED_CIP",
                    "cip_effectiveness": float(row.get("cip_effectiveness", 0.85)),
                    "di_before": di_actual,
                    "di_after": 0.0,
                    "details": f"CIP Cycle {int(features.get('cip_cycle', 1))} executed"
                }

            # 10. Persist to database
            derived_data = {
                "exchanger_id": self.exchanger_id,
                "scenario_id": self.scenario_id,
                "time_h": time_h,
                "timestamp_utc": now_utc,
                "di_fouling_actual": di_actual,
                "di_fouling_predicted": di_predicted,
                "di_total": di_total,
                "u_retention": u_ret,
                "q_retention": q_ret,
                "thermal_efficiency": th_eff,
                "performance_score": perf_score,
                "fouling_risk": risk_res.fouling_risk,
                "maintenance_risk": risk_res.maintenance_risk,
                "maintenance_state": risk_res.maintenance_state.value,
                "degradation_rate": risk_res.degradation_rate,
                "ttt_warning_h": risk_res.ttt_warning_h,
                "ttt_critical_h": risk_res.ttt_critical_h,
                "recommendation_state": clean_res.state.value,
                "recommendation_text": clean_res.text,
                "consecutive_above_warn": clean_res.consecutive_above_warn,
                "consecutive_above_crit": clean_res.consecutive_above_crit
            }

            await self.db.insert_cycle_data(
                raw_reading=row,
                derived=derived_data,
                alerts=alerts_list,
                cip_event=cip_record
            )

            # 11. Assemble comprehensive real-time payload
            packet = {
                "type": "TELEMETRY_UPDATE",
                "exchanger_id": self.exchanger_id,
                "scenario_id": self.scenario_id,
                "time_h": time_h,
                "timestamp_utc": now_utc,
                "cursor_index": self.cursor_index,
                "total_rows": self._total_rows,
                "sensors": {
                    "T_in_measured_C": round(T_meas, 2),
                    "m_dot_measured_kg_s": round(float(row.get("m_dot_measured_kg_s", 0)), 3),
                    "Re": round(float(row.get("Re", 0)), 1),
                    "u_m_s": round(float(row.get("u_m_s", 0)), 4),
                    "tau_w_Pa": round(float(row.get("tau_w_Pa", 0)), 4),
                    "dP_Pa": round(float(row.get("dP_Pa", 0)), 2),
                    "Rf_m2K_W": float(row.get("Rf_m2K_W", 0)),
                    "U_clean_W_m2K": round(U_clean, 2),
                    "U_overall_W_m2K": round(U_overall, 2),
                    "Q_W": round(Q_W, 1),
                    "Q_clean_W": round(Q_clean, 1),
                    "thermal_efficiency": round(th_eff, 4),
                    "rho_kg_m3": round(rho_calc, 2),
                    "mu_Pa_s": float(f"{mu_calc:.6e}"),
                },
                "physical": {
                    "di_fouling_actual": di_actual,
                    "di_fouling_predicted": di_predicted,
                    "di_total": di_total,
                    "u_retention": round(u_ret, 5),
                    "q_retention": round(q_ret, 5),
                    "performance_score": round(perf_score, 1),
                },
                "risk": {
                    "fouling_risk": round(risk_res.fouling_risk, 1),
                    "maintenance_risk": round(risk_res.maintenance_risk, 1),
                    "maintenance_state": risk_res.maintenance_state.value,
                    "degradation_rate": risk_res.degradation_rate,
                    "ttt_warning_h": risk_res.ttt_warning_h,
                    "ttt_critical_h": risk_res.ttt_critical_h,
                },
                "recommendation": {
                    "state": clean_res.state.value,
                    "text": clean_res.text,
                    "consecutive_above_warn": clean_res.consecutive_above_warn,
                    "consecutive_above_crit": clean_res.consecutive_above_crit,
                },
                "thresholds": {
                    "warning": settings.di_warning,
                    "critical": settings.di_critical,
                },
                "new_alerts": alerts_list,
                "cip_event": cip_record is not None
            }

            # Update worker state
            self.last_reading_time_utc = now_utc
            self.last_time_h = time_h
            self.last_state_packet = packet
            self.records_processed += 1
            self.cursor_index += 1

            # Persist cursor state periodically (every 50 rows or at alert)
            if self.records_processed % 50 == 0 or alerts_list:
                await self.db.save_ingestion_state(
                    self.exchanger_id, self.scenario_id, self.cursor_index,
                    self.last_time_h, self.records_processed
                )

            return packet

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.exception("Error processing row %d (time_h: %s): %s",
                             self.cursor_index, row.get("time_h"), e)
            self.cursor_index += 1
            return None

    # ── Background Loop Execution ───────────────────────────────────────

    async def run(self) -> None:
        """Main autonomous execution loop."""
        self.is_running = True
        logger.info("Ingestion worker started for exchanger %s (Interval: %.2fs)",
                    self.exchanger_id, self.replay_interval)

        while self.is_running:
            if not self.is_paused:
                packet = await self.process_next_record()
                if packet:
                    await self.broadcast_state(packet)

            await asyncio.sleep(self.replay_interval)

        logger.info("Ingestion worker stopped.")

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False

    def set_speed(self, interval_seconds: float) -> None:
        self.replay_interval = max(0.01, float(interval_seconds))

    def reset_cursor(self) -> None:
        self.cursor_index = 0
        self.fe = FeatureEngineering()
        self.risk_engine.reset_history()
